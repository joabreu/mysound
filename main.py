"""Main module for mysound recommender."""

import os
from copy import deepcopy
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import requests
import spotipy
from datasets import load_dataset
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from musicbrainzngs import musicbrainz
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

USER_RECENT = 10
USER_GLOBAL = 50
ARTIST_SIMILAR = 5
ARTIST_SIMILAR_RECS = 15
SIM_THRESHOLD = 0.60
MAX_NEW = 50

load_dotenv()

SCOPES = ["user-top-read", "playlist-modify-private", "user-read-recently-played", "playlist-modify-public"]

musicbrainz.set_useragent("mysound", "0.1", "mysound@domain.com")
musicbrainz.set_rate_limit(limit_or_interval=15.0, new_requests=9)


def sp_client() -> Any:
    """Create and return a Spotify client."""
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            scope=" ".join(SCOPES),
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            cache_path=".cache-spotify",
        ),
        requests_timeout=10,
        retries=10,
        backoff_factor=0.3,
    )


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
    return list(set(tags))


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


@musicbrainz._rate_limit  # pylint:disable=protected-access
def find_sptrack(track: str, artist: str) -> Tuple[str | None, str | None, float]:
    """Return a track Spotify given track name and artist name."""
    query = "artist:" + artist + " track:" + track
    t = sp.search(q=query[: min(250, len(query))], limit=1, type="track")
    if len(t["tracks"]["items"]):
        sp_track = t["tracks"]["items"][0]
        sp_name = sp_track["name"]
        for a in sp_track["artists"]:
            if fuzz.partial_ratio(a, artist) > 90 and fuzz.partial_ratio(sp_name, track) > 90:
                return a["id"], sp_track["uri"], np.log1p(sp_track["popularity"])
    return None, None, 0.0


@musicbrainz._rate_limit  # pylint:disable=protected-access
def find_sptags(artist_id: str) -> List:
    """Return a genre for Spotify artist given artist ID."""
    a = sp.artist(artist_id)
    return a["genres"] if a is not None else [""]


def add_artist_genres_and_tracks(artist_name: str, releases: dict, prev_tags: List | None = None) -> List:
    """Add artist genres and tracks."""
    tracks = []
    sp_tags_artist = None
    for r in releases["release-list"]:
        if "id" in r:
            t = musicbrainz.browse_recordings(release=r["id"], includes=["tags"])
        else:
            t = r

        for t_1 in t["recording-list"]:
            tags: List = []
            if prev_tags is not None:
                tags = tags + prev_tags

            try:
                aid, uri, sp_rank = find_sptrack(track=t_1["title"], artist=artist_name)
                if aid is not None and sp_tags_artist is None:
                    sp_tags_artist = find_sptags(artist_id=aid)
            except spotipy.exceptions.SpotifyException:
                uri, sp_rank = None, 0.0
                sp_tags_artist = None

            tags = tags + deezer_track_description_from_name(artist_name, t_1["title"])
            if sp_tags_artist is not None:
                tags = tags + sp_tags_artist
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


def get_artist_tracks(tracks: dict, artist: dict, track_name: str | None = None, limit: int = 10) -> None:
    """Get single artist tracks."""
    if track_name is None:
        releases = musicbrainz.browse_recordings(artist=artist["id"], includes=["tags"], limit=limit)
    else:
        releases = musicbrainz.search_recordings(artist=artist["name"], recording=track_name, limit=1)

    releases_tags = order_filter_tags(artist.get("tag-list", []))
    releases_tags = releases_tags + order_filter_tags(releases.get("tag-list", []))

    releases = {"release-list": [releases]}

    tracks[artist["name"]] = {
        "id": artist["id"],
        "releases": releases,
        "genres": releases_tags,
        "tracks": add_artist_genres_and_tracks(artist["name"], releases, prev_tags=releases_tags),
    }

    # Update with track info
    full_tags = []
    for g in tracks[artist["name"]]["tracks"]:
        full_tags.extend(g["tags"])
    tracks[artist["name"]]["genres"] = list(set(tracks[artist["name"]]["genres"] + full_tags))
    print(artist["name"], tracks[artist["name"]]["genres"])


def get_artist_top_tracks(tracks: dict, artist_name: str, track_name: str | None = None, limit: int = 10) -> None:
    """Get top and similar tracks for artist."""
    if artist_name not in tracks:
        result = musicbrainz.search_artists(artist=artist_name, limit=1, strict=True)
        if len(result["artist-list"]) > 0:
            artist = result["artist-list"][0]
        else:
            return

        get_artist_tracks(tracks, artist, track_name, limit=1)
    elif limit >= 1:
        artist = tracks[artist_name]
        if len(artist["genres"]) == 0:
            return
        result = musicbrainz.search_artists(tag=artist["genres"], limit=limit, offset=0)
        for r in result["artist-list"]:
            if r["name"] in tracks:
                continue
            try:
                similar_artist = musicbrainz.search_artists(artist=r["name"], limit=1, strict=True)
                for a in similar_artist["artist-list"]:
                    get_artist_tracks(tracks, a, None, ARTIST_SIMILAR_RECS)
            except musicbrainz.MusicBrainzError:
                continue


def track_description(mb_genres: List | None) -> str:
    """Build description string enriched with MusicBrainz genres."""
    return ", ".join(mb_genres) if mb_genres is not None else ""


def get_top_tracks(limit_r: int = 10, limit_t: int = 10) -> dict:
    """Get current user recently played tracks and top tracks."""
    top = [t["track"] for t in sp.current_user_recently_played(limit=limit_r)["items"]]
    top.extend(sp.current_user_top_tracks(limit=limit_r, time_range="short_term")["items"])
    top.extend(sp.current_user_top_tracks(limit=limit_t, time_range="long_term")["items"])
    top.extend(sp.current_user_top_artists(limit=limit_t, time_range="long_term")["items"])

    user_tracks: dict = {}
    for f in tqdm(top):
        if "artists" in f:
            track = f["name"]
            for artist in f["artists"]:
                if artist["name"] in user_tracks:
                    continue
                get_artist_top_tracks(
                    user_tracks,
                    artist_name=artist["name"],
                    track_name=track,
                    limit=1,
                )
        else:
            if f["name"] in user_tracks:
                continue
            get_artist_top_tracks(user_tracks, artist_name=f["name"], track_name=None, limit=ARTIST_SIMILAR_RECS)

    return user_tracks


def embed_tags(tags: List, lookup: dict, dim: int) -> np.array:
    """Compute vectors for each tag."""
    vectors = [lookup[t] for t in tags if t in lookup]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)


def generate_recommends(top_tracks: dict, latest_tracks: dict) -> List:
    """Generate recommendations using Tfid vectorizer."""
    cand_descs = []
    tracks_descs = []
    for a in tqdm(latest_tracks.keys()):
        for t in latest_tracks[a]["tracks"]:
            cand_descs.append(track_description(t["tags"]))
            tracks_descs.append((a, t["name"], t["tags"], t["uri"], t["rank"]))

    top_indices = [i for i, (a, _, _, _, _) in enumerate(tracks_descs) if a in top_tracks]
    candidate_indices = [i for i in range(len(tracks_descs)) if i not in top_indices]
    tracks_descs = [t for i, t in enumerate(tracks_descs) if i in candidate_indices]
    ranks = [r for i, (_, _, _, _, r) in enumerate(tracks_descs) if i in candidate_indices]

    vectorizer = TfidfVectorizer(
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+[^,]+\b",
        ngram_range=(1, 1),
        use_idf=True,
        min_df=0.10,
    )

    X = vectorizer.fit_transform(cand_descs)
    embedding_dim = len(next(iter(mwe_lookup.values())))
    track_embeddings = csr_matrix(
        np.array(
            [
                embed_tags(track["tags"], mwe_lookup, embedding_dim)
                for artist in latest_tracks.values()
                for track in artist["tracks"]
            ]
        )
    )

    X = normalize(X, norm="l2")
    track_embeddings = normalize(track_embeddings, norm="l2")
    hybrid_matrix = hstack([X, track_embeddings]).tocsr()
    sims = cosine_similarity(
        # np.median(hybrid_matrix[top_indices].toarray(), axis=0),
        hybrid_matrix[top_indices].max(axis=0),
        hybrid_matrix[candidate_indices],
    ).ravel()
    print(len(tracks_descs), len(top_indices), len(candidate_indices), len(sims), len(tracks_descs))
    return sorted(zip(tracks_descs, sims, ranks), key=lambda p: (p[1], p[2]), reverse=True)


def create_playlist(recommended: List) -> None:
    """Create new playlist given recommended tracks."""
    user_name = sp.current_user()["uri"].split(":")[2]
    if len(recommended):
        playlist_date = datetime.now().strftime("%b/%-d")
        playlist_name = f"Recommended by MySound ({playlist_date})"
        playlist_id = sp.user_playlist_create(user=user_name, name=playlist_name)["id"]
        for uri in recommended:
            sp.user_playlist_add_tracks(user=user_name, playlist_id=playlist_id, tracks=[uri])


def recommend() -> None:
    """Generate recommendations for user."""
    tracks = {}
    user_tracks = get_top_tracks(limit_r=USER_RECENT, limit_t=USER_GLOBAL)
    latest_tracks = deepcopy(user_tracks)
    for k in tqdm(user_tracks.keys()):
        get_artist_top_tracks(latest_tracks, artist_name=k, limit=ARTIST_SIMILAR)

    ranked = generate_recommends(user_tracks, latest_tracks)

    print("Top recommendations:")
    recommended = []
    for i, ((artist, track, d, uri, rank), score, _) in enumerate(ranked, start=0):
        if artist in tracks:
            continue

        tracks[artist] = [track]
        if score >= SIM_THRESHOLD:
            if uri is not None:
                print(f"{i:2d}. {artist} — {track}  (rank={rank:.2f}, sim={score:.2f}, d='{d}')")
                recommended.append(uri)
        if len(recommended) >= MAX_NEW:
            break

    create_playlist(recommended)


sp = sp_client()
ds = load_dataset("seungheondoh/musical-word-embedding", split="tag")
mwe_lookup = {row["token"]: np.array(row["vector"]) for row in ds}

if __name__ == "__main__":
    recommend()
