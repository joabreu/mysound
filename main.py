"""Main module for mysound recommender."""

import functools
import json
import os
import time
from datetime import datetime
from random import shuffle
from typing import Any, List, Tuple

import numpy as np
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from musicbrainzngs import musicbrainz
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from ytmusicapi import YTMusic
from ytmusicapi.exceptions import YTMusicServerError

USER_RECENT = 5
USER_GLOBAL = 10
ARTIST_SIMILAR = 5
ARTIST_SIMILAR_RECS = None  # To fetch all tracks
SIM_THRESHOLD = 0.40
MAX_NEW = 50

load_dotenv()

CACHE_FILE = ".cache.json"

SCOPES = ["user-top-read", "playlist-modify-private", "user-read-recently-played", "playlist-modify-public"]
PLAYLIST_PREFIX = "Recommended by MySound"

musicbrainz.set_useragent("mysound", "0.1", "mysound@domain.com")
musicbrainz.set_rate_limit(limit_or_interval=1.0, new_requests=1)


def retry(
    exceptions: Tuple,
    max_attempts: int = 10,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Any:
    """Retry decorator for a function."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts:
                        print("ERROR: max attempts reached")
                        return None
                    time.sleep(delay)
                    delay *= backoff_factor
            return None

        return wrapper

    return decorator


def load_cache() -> dict:
    """Load already found artists and tracks."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    return cache


def save_cache(cache: dict) -> None:
    """Save already found artists and tracks for future usage."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def order_filter_tags(tag_list: List, prev_list: List | None = None, token: str = "name", limit: int = 0) -> List:
    """Order and filter music tags."""
    tags = list(set(sorted([(t[token], t["count"]) for t in tag_list], key=lambda p: p[1], reverse=True)))
    if limit > 0:
        tags = [t[0] for t in tags[: min(len(tags), limit)]]
    else:
        tags = [t[0] for t in tags]
    if prev_list is not None:
        tags = prev_list + tags
    tags = [t for t in tags if len(t)]
    return tags


def deezer_track_description_from_name(artist_name: str, track_name: str) -> List:
    """Search Deezer for a track by artist and title."""
    query = f"{artist_name} {track_name}"
    url = "https://api.deezer.com/search"
    r = requests.get(url, params={"q": query, "limit": "1"})

    if r.status_code != 200:
        return []

    results = r.json().get("data", [])
    if not results:
        return []

    # Try to fetch genre from album/artist (Deezer doesn’t give track-level genres)
    genre = ""
    track = results[0]
    artist_id = track.get("artist", {}).get("id")
    if artist_id:
        artist_info = requests.get(f"https://api.deezer.com/artist/{artist_id}")
        if artist_info.status_code == 200:
            genre = artist_info.json().get("genre_id", "")

    return [
        str(genre),
    ]


def find_yttrack(track: str, artist: str) -> Tuple[str | None, str | None, float]:
    """Return YouTube Music videoId and pseudo-popularity."""
    query = f"{track} {artist}"
    try:
        results = yt.search(query, filter="songs", limit=1)
    except json.JSONDecodeError:
        results = []

    for item in results:
        title = item.get("title", "")
        artists = " ".join(a["name"] for a in item.get("artists", []))

        if (
            fuzz.partial_ratio(title.lower(), track.lower()) > 90
            and fuzz.partial_ratio(artists.lower(), artist.lower()) > 90
        ):
            video_id = item["videoId"]
            return "", video_id, 0.0  # score not available in YT
    return None, None, 0.0


def get_blacklist() -> list[str]:
    """Get list of already recommended tracks."""
    playlists = yt.get_library_playlists(limit=None)
    blacklist = []
    for p in playlists:
        if p["title"].startswith(PLAYLIST_PREFIX):
            try:
                tracks = yt.get_playlist(p["playlistId"], limit=None)
            except json.JSONDecodeError:
                tracks = {"tracks": []}
            for t in tracks["tracks"]:
                if "videoId" in t:
                    blacklist.append(t["videoId"])
    return blacklist


def add_artist_genres_and_tracks(artist_name: str, releases: dict, prev_tags: List | None = None) -> List:
    """Add artist genres and tracks."""
    tracks = []
    deezer_tags_artist = None
    for r in releases["release-list"]:
        if "id" in r:
            t = musicbrainz.browse_recordings(release=r["id"], includes=["tags"])
        else:
            t = r

        for t_1 in t["recording-list"]:
            tags: List = []
            if prev_tags is not None:
                tags = prev_tags + tags
            if deezer_tags_artist is None:
                deezer_tags_artist = deezer_track_description_from_name(artist_name, t_1["title"])
            tags = deezer_tags_artist + tags

            _, uri, sp_rank = find_yttrack(track=t_1["title"], artist=artist_name)
            tracks.append(
                {
                    "name": t_1["title"],
                    "artists": [{"name": artist_name}],
                    "tags": order_filter_tags(t_1.get("tag-list", []), prev_list=tags),
                    "uri": uri,
                    "rank": sp_rank,
                }
            )
    return tracks


@retry(
    exceptions=(musicbrainz.NetworkError,),
)
def get_artist_tracks(
    tracks: dict, artist: dict, track_name: str | None = None, limit: int | None = 10, w: int = 1
) -> None:
    """Get single artist tracks."""
    releases_tags = order_filter_tags(artist.get("tag-list", []))

    if track_name is not None:
        track_releases = musicbrainz.search_recordings(artist=artist["name"], recording=track_name, limit=1)
        releases_tags = releases_tags + order_filter_tags(track_releases.get("tag-list", []))
        releases = {"release-list": [track_releases]}
    else:
        releases = musicbrainz.browse_recordings(artist=artist["id"], includes=["tags"], limit=limit)
        releases_tags = releases_tags + order_filter_tags(releases.get("tag-list", []))
        releases = {"release-list": [releases]}

    artist_tracks = add_artist_genres_and_tracks(artist["name"], releases, prev_tags=releases_tags)
    if artist["name"] in tracks:
        a = tracks[artist["name"]]
        releases_tags.extend(a["genres"])
        artist_tracks.extend(a["tracks"])

    tracks[artist["name"]] = {
        "id": artist["id"],
        "releases": releases,
        "genres": releases_tags,
        "tracks": artist_tracks,
        "w": w,
    }

    # Update with track info
    full_tags = []
    for g in tracks[artist["name"]]["tracks"]:
        full_tags.extend(g["tags"])
    tracks[artist["name"]]["genres"] = list(set(tracks[artist["name"]]["genres"] + full_tags))
    print(artist["name"], tracks[artist["name"]]["genres"])


def get_similar_artist_tracks(tracks: dict, artists: dict) -> None:
    """Get similar artist tracks."""
    cache = load_cache()

    for r in artists["artist-list"]:
        if r["name"] in tracks:
            continue
        if r["name"] in cache:
            tracks[r["name"]] = cache[r["name"]]
            tracks[r["name"]]["w"] = 1
            continue
        get_artist_tracks(tracks, r, None, ARTIST_SIMILAR_RECS, 1)
        cache[r["name"]] = tracks[r["name"]]

    save_cache(cache)


@retry(
    exceptions=(musicbrainz.NetworkError,),
)
def get_artist_top_tracks(
    tracks: dict, artist_name: str, track_name: str | None = None, limit: int | None = 10, w: int = 1
) -> None:
    """Get top and similar tracks for artist."""
    result = musicbrainz.search_artists(artist=artist_name, limit=1, strict=True)
    if len(result["artist-list"]) > 0:
        artist = result["artist-list"][0]
    else:
        return

    get_artist_tracks(tracks, artist, track_name, limit=limit, w=w)


def track_description(mb_genres: List | None) -> str:
    """Build description string enriched with MusicBrainz genres."""
    return ", ".join(mb_genres) if mb_genres is not None else ""


def get_top_tracks(limit_r: int = 10, limit_t: int = 10) -> dict:
    """Get current user recently played tracks and top tracks."""
    top = []

    # Recently played
    history = yt.get_history()
    for item in history[:limit_r]:
        if "videoId" in item:
            top.append((item, limit_r))

    # Library songs (top)
    library = yt.get_library_artists(limit=limit_t, order="recently_added")
    for item in library[:limit_t]:
        top.append((item, limit_t))

    user_tracks: dict = {}
    for f, w in tqdm(top):
        if "artists" in f:
            track = f["title"]
            for artist in f["artists"]:
                if artist["name"] in user_tracks:
                    continue
                get_artist_top_tracks(
                    user_tracks,
                    artist_name=artist["name"],
                    track_name=track,
                    limit=1,
                    w=w,
                )
        else:
            if f["artist"] in user_tracks:
                continue
            get_artist_top_tracks(user_tracks, artist_name=f["artist"], track_name=None, limit=1, w=w)

    return user_tracks


def embed_tags(tags: List, lookup: dict, dim: int) -> np.array:
    """Compute vectors for each tag."""
    vectors = [lookup[t] for t in tags if t in lookup]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)


def generate_recommends(top_tracks: dict, latest_tracks: dict) -> List:
    """Generate recommendations using Tfid vectorizer."""
    cand_descs = []
    tracks_descs = []
    embed_descs = []
    tracks_weights = []

    for a in tqdm(top_tracks.keys()):
        for t in top_tracks[a]["tracks"]:
            cand_descs.append(track_description(t["tags"]))
            tracks_descs.append((a, t["name"], t["tags"], t["uri"], t["rank"]))
            embed_descs.append(embed_tags(t["tags"], mwe_lookup, embedding_dim))
            tracks_weights.append(top_tracks[a]["w"])
    for a in tqdm(latest_tracks.keys()):
        # Shuffle artist tracks
        shuffle(latest_tracks[a]["tracks"])

        for t in latest_tracks[a]["tracks"]:
            cand_descs.append(track_description(t["tags"]))
            tracks_descs.append((a, t["name"], t["tags"], t["uri"], t["rank"]))
            embed_descs.append(embed_tags(t["tags"], mwe_lookup, embedding_dim))
            tracks_weights.append(latest_tracks[a]["w"])

    top_indices = [
        i
        for i, (a, t, _, _, _) in enumerate(tracks_descs)
        for artist, artist_data in top_tracks.items()
        for track in artist_data["tracks"]
        if a == artist and t == track["name"]
    ]
    candidate_indices = [i for i in range(len(tracks_descs)) if i not in top_indices]
    tracks_descs = [t for i, t in enumerate(tracks_descs) if i in candidate_indices]
    ranks = [r for i, (_, _, _, _, r) in enumerate(tracks_descs) if i in candidate_indices]
    tracks_weights = np.array(tracks_weights)

    vectorizer = TfidfVectorizer(
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+[^,]+\b",
        ngram_range=(1, 1),
        use_idf=True,
        min_df=0.02,
    )

    X = vectorizer.fit_transform(cand_descs)
    track_embeddings = csr_matrix(np.array(embed_descs)).toarray()

    X = np.hstack([X.toarray(), track_embeddings.max(axis=1).reshape(-1, 1)])
    sims = cosine_similarity(
        np.average(X[top_indices], axis=0, weights=tracks_weights[top_indices]).reshape(1, -1),
        X[candidate_indices],
    ).ravel()
    print(len(tracks_descs), len(top_indices), len(candidate_indices), len(sims), len(tracks_descs))
    return sorted(zip(tracks_descs, sims, ranks), key=lambda p: p[1], reverse=True)


@retry(
    exceptions=(YTMusicServerError,),
)
def add_to_playlist(playlist_id: str, track: str) -> None:
    """Add music to playlist."""
    yt.add_playlist_items(playlist_id, [track], duplicates=True)


def create_playlist(recommended: List) -> None:
    """Create new playlist given recommended tracks."""
    if len(recommended):
        playlist_date = datetime.now().strftime("%b/%-d")
        playlist_name = f"{PLAYLIST_PREFIX} ({playlist_date})"
        playlist_id = yt.create_playlist(playlist_name, "Created by mySound")
        for track in recommended:
            add_to_playlist(playlist_id, track)


@retry(
    exceptions=(musicbrainz.NetworkError,),
)
def find_similar_artists(g: str, artists: dict) -> None:
    """Find artists with genre 'g'."""
    artists["artist-list"].extend(musicbrainz.search_artists(tag=g, limit=ARTIST_SIMILAR, offset=None)["artist-list"])


def recommend() -> None:
    """Generate recommendations for user."""
    tracks = {}
    blacklist = get_blacklist()
    user_tracks = get_top_tracks(limit_r=USER_RECENT, limit_t=USER_GLOBAL)
    latest_tracks: dict = {}
    for k in tqdm(user_tracks.keys()):
        if len(user_tracks[k]["genres"]) == 0:
            continue
        artists: dict = {"artist-list": []}
        for g in user_tracks[k]["genres"]:
            find_similar_artists(g, artists)
        artists["artist-list"].append({"id": user_tracks[k]["id"], "name": k})
        get_similar_artist_tracks(latest_tracks, artists)

    ranked = generate_recommends(user_tracks, latest_tracks)
    shuffle(ranked)

    print("Top recommendations:")
    for i, ((artist, track, d, uri, rank), score, _) in enumerate(ranked, start=0):
        if artist in tracks:
            continue

        if score >= SIM_THRESHOLD and uri is not None and uri not in blacklist:
            print(f"{i:2d}. {artist} — {track}  (rank={rank:.2f}, sim={score:.2f}, d='{d}')")
            tracks[artist] = {"track": track, "uri": uri}
        if len(tracks) >= MAX_NEW:
            break

    create_playlist([t["uri"] for _, t in tracks.items()])


yt = YTMusic(".headers-auth.json")
ds = load_dataset("seungheondoh/musical-word-embedding", split="tag")
mwe_lookup = {row["token"]: np.array(row["vector"]) for row in ds}
embedding_dim = len(next(iter(mwe_lookup.values())))

if __name__ == "__main__":
    recommend()
