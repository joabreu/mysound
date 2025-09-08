"""Main module for mysound recommender."""

import os
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
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

USER_RECENT = 50
USER_GLOBAL = 20
ARTIST_SIMILAR = 15
ARTIST_SIMILAR_RECS = 10
SIM_THRESHOLD = 0.80
MAX_NEW = 50

load_dotenv()

SCOPES = ["user-top-read", "playlist-modify-private", "user-read-recently-played", "playlist-modify-public"]

musicbrainz.set_useragent("mysound", "0.1", "mysound@domain.com")
musicbrainz.set_rate_limit(limit_or_interval=1.0, new_requests=3)


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


def order_filter_tags(tag_list: List, prev_list: List | None = None, token: str = "name") -> List:
    """Order and filter music tags."""
    tags = list(set(sorted([(t[token], t["count"]) for t in tag_list], key=lambda p: p[1], reverse=True)))
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
    nb_fan = 0
    track = results[0]
    artist_id = track.get("artist", {}).get("id")
    rank = int(track.get("rank", 0))
    if artist_id:
        artist_info = requests.get(f"https://api.deezer.com/artist/{artist_id}")
        if artist_info.status_code == 200:
            genre = artist_info.json().get("genre_id", "")
            nb_fan = int(artist_info.json().get("id", 0))

    return [
        f"fans_{int(np.log1p(nb_fan))}",
        f"rank_{int(np.log1p(rank))}",
        str(genre),
    ]


@musicbrainz._rate_limit  # noqa: W0212 # pylint:disable=protected-access
def find_sptrack(track: str, artist: str) -> Tuple[str | None, str | None, List]:
    """Return a track Spotify given track name and artist name."""
    query = "artist:" + artist + " track:" + track
    t = sp.search(q=query[: min(250, len(query))], limit=1, type="track")
    if len(t["tracks"]["items"]):
        sp_track = t["tracks"]["items"][0]
        sp_name = sp_track["name"]
        for a in sp_track["artists"]:
            if fuzz.partial_ratio(a, artist) > 90 and fuzz.partial_ratio(sp_name, track) > 90:
                return a["id"], sp_track["uri"], [f"popularity_{int(np.log1p(sp_track['popularity']))}"]
    return None, None, [""]


@musicbrainz._rate_limit  # noqa: W0212 # pylint:disable=protected-access
def find_sptags(artist_id: str) -> List:
    """Return a genre for Spotify artist given artist ID."""
    a = sp.artist(artist_id)
    return a["genres"] if a is not None else [""]


def add_artist_genres_and_tracks(artist_name: str, releases: dict, prev_tags: List | None = None) -> List:
    """Add artist genres and tracks."""
    tracks = []
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
                aid, uri, sp_tags = find_sptrack(track=t_1["title"], artist=artist_name)
            except spotipy.exceptions.SpotifyException:
                aid, uri, sp_tags = None, None, [""]

            try:
                if aid is not None:
                    sp_tags = sp_tags + find_sptags(artist_id=aid)
            except spotipy.exceptions.SpotifyException:
                pass

            tags = tags + deezer_track_description_from_name(artist_name, t_1["title"])
            tags = tags + sp_tags
            tracks.append(
                {
                    "name": t_1["title"],
                    "artists": [{"name": artist_name}],
                    "tags": order_filter_tags(t_1.get("tag-list", []), prev_list=tags),
                    "uri": uri,
                }
            )
    return tracks


def get_artist_tracks(tracks: dict, artist: dict, track_name: str | None = None, limit: int = 10) -> None:
    """Get single artist tracks."""
    if track_name is None:
        releases = musicbrainz.browse_recordings(artist=artist["id"], includes=["tags"], limit=limit)
    else:
        releases = musicbrainz.search_recordings(artist=artist["name"], recording=track_name, limit=limit)

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
    if artist_name not in tracks.keys():
        result = musicbrainz.search_artists(artist=artist_name, limit=1, strict=True)
        if len(result["artist-list"]) > 0:
            artist = result["artist-list"][0]
        else:
            return

        get_artist_tracks(tracks, artist, track_name, limit)
    elif limit > 1:
        artist = tracks[artist_name]
        result = musicbrainz.search_artists(tag=artist["genres"], limit=limit, offset=0)
        for r in result["artist-list"]:
            if r["name"] in tracks.keys():
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


def get_top_tracks(limit_r: int = 10, limit_t: int = 10) -> List:
    """Get current user recently played tracks and top tracks."""
    top = [t["track"] for t in sp.current_user_recently_played(limit=limit_r)["items"]]
    top.extend(sp.current_user_top_tracks(limit=limit_r, time_range="short_term")["items"])
    top.extend(sp.current_user_top_tracks(limit=limit_t, time_range="long_term")["items"])
    return top


def find_uri(track: str, artist: str) -> str | None:
    """Return a track Spotify URI given track name and artist name."""
    t = sp.search(q="artist:" + artist + " track:" + track, limit=1, type="track")
    if len(t["tracks"]["items"]):
        sp_track = t["tracks"]["items"][0]
        sp_artist = sp_track["artists"][0]["name"]
        sp_name = sp_track["name"]
        if fuzz.partial_ratio(sp_artist, artist) > 90 and fuzz.partial_ratio(sp_name, track) > 90:
            return sp_track["uri"]
    return None


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
            tracks_descs.append((t["artists"][0]["name"], t["name"], t["tags"], t["uri"]))

    top_indices = [i for i, (a, _, _, _) in enumerate(tracks_descs) if a in top_tracks.keys()]
    candidate_indices = [i for i in range(len(tracks_descs)) if i not in top_indices]
    tracks_descs = [t for i, t in enumerate(tracks_descs) if i in candidate_indices]

    vectorizer = TfidfVectorizer(
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+[^,]+\b",
        ngram_range=(1, 3),
        use_idf=False,
        min_df=0.05,
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

    hybrid_matrix = hstack([X, track_embeddings.mean(axis=1)]).tocsr()
    sims = cosine_similarity(
        np.asarray(hybrid_matrix[top_indices].mean(axis=0)), hybrid_matrix[candidate_indices]
    ).ravel()
    print(len(sims), len(tracks_descs))
    ranked = sorted(zip(tracks_descs, sims), key=lambda p: p[1], reverse=True)
    return ranked


def create_playlist(recommended: List) -> None:
    """Create new playlist given recommended tracks."""
    user_name = sp.current_user()["uri"].split(":")[2]
    if len(recommended):
        playlist_date = datetime.now().strftime("%b/%-d")
        playlist_name = f"Recommended by MySound ({playlist_date})"
        playlist_id = sp.user_playlist_create(user=user_name, name=playlist_name)["id"]
        for uri in recommended:
            sp.user_playlist_add_tracks(user=user_name, playlist_id=playlist_id, tracks=[uri])


def pick_from_sims(ranked: List) -> List:
    """Pick random track from candidates that have a given sim."""
    sims = []
    for _, score in ranked:
        sims.append(score)
    sims = np.array(sims)

    ranked_new = []
    for _, score in ranked:
        matching_indices = np.where(sims == score)[0]
        chosen_idx = np.random.choice(matching_indices)
        ranked_new.append(ranked[chosen_idx])
    return ranked_new


def recommend() -> None:
    """Generate recommendations for user."""
    tracks = {}

    user_tracks: dict = {}
    top_tracks_user = get_top_tracks(limit_r=USER_RECENT, limit_t=USER_GLOBAL)
    for f in tqdm(top_tracks_user):
        track = f["name"]
        for artist in f["artists"]:
            get_artist_top_tracks(user_tracks, artist_name=artist["name"], track_name=track, limit=1)
            tracks[artist["name"]] = [track]

    latest_tracks = user_tracks.copy()
    for k in tqdm(user_tracks.keys()):
        get_artist_top_tracks(latest_tracks, artist_name=k, limit=ARTIST_SIMILAR)

    ranked = generate_recommends(user_tracks, latest_tracks)
    ranked = pick_from_sims(ranked)

    print("Top recommendations:")
    recommended = []
    for i, ((artist, track, d, uri), score) in enumerate(ranked, start=0):
        if artist in tracks:
            continue

        tracks[artist] = [track]
        if score >= SIM_THRESHOLD:
            if uri is not None:
                print(f"{i:2d}. {artist} — {track}  (sim={score:.2f}, d='{d}')")
                recommended.append(uri)
        if len(recommended) >= MAX_NEW:
            break

    create_playlist(recommended)


sp = sp_client()
ds = load_dataset("seungheondoh/musical-word-embedding", split="tag")
mwe_lookup = {row["token"]: np.array(row["vector"]) for row in ds}

if __name__ == "__main__":
    recommend()
