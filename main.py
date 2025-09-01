"""Main module for mysound recommender."""

import os
from typing import Any, Dict, List

import musicbrainzngs
import numpy as np
import requests
import spotipy
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

USER_RECENT = 10
USER_GLOBAL = 50
ARTIST_SIMILAR = 5
SIM_THRESHOLD = 0.28

load_dotenv()

LB_HEADERS = {
    "User-Agent": f"MyApp/1.0 (ID={os.getenv('LB_CLIENT_ID')})",
    "Authorization": f"Token {os.getenv('LB_CLIENT_SECRET')}",
}

SCOPES = ["user-top-read", "playlist-modify-private", "user-read-recently-played", "playlist-modify-public"]


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


def mb_client() -> None:
    """Set the MusicBrainz API."""
    musicbrainzngs.set_useragent("spotify-recommender-demo", "0.1", "youremail@example.com")
    musicbrainzngs.auth(os.getenv("MB_USER"), os.getenv("MB_PASSWORD"))
    musicbrainzngs.set_rate_limit(False)


def mb_artist_genres(artist_name: str) -> List | None:
    """Fetch MusicBrainz tags/genres for an artist name."""
    result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
    if result["artist-list"]:
        mbid = result["artist-list"][0]["id"]
        artist = musicbrainzngs.get_artist_by_id(mbid, includes=["tags", "ratings"])
        tags = [t["name"] for t in artist["artist"].get("tag-list", [])]
        tags.extend([t["rating"] for t in artist["artist"].get("ratings", [])])
        return tags
    return None


def get_artist_top_tracks(artist_name: str, limit: int = 10) -> List:
    """Get top and similar tracks for artist."""
    result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
    if result["artist-list"]:
        mbid = result["artist-list"][0]["id"]
    else:
        return []

    url = f"https://api.listenbrainz.org/1/lb-radio/artist/{mbid}"
    params: Dict[str, Any] = {
        "mode": "easy",
        "pop_begin": 0,
        "pop_end": 100,
        "max_similar_artists": limit,
        "max_recordings_per_artist": 1,
    }
    resp = requests.get(url, params=params, headers=LB_HEADERS)

    data = []
    for r in resp.json().values():
        data.append(
            {
                "similar_artist_mbid": r[0]["similar_artist_mbid"],
                "name": r[0]["similar_artist_name"],
            }
        )

    tracks = []
    mb_genres = mb_artist_genres(artist_name)
    result = musicbrainzngs.search_artists(tag=mb_genres, limit=limit, offset=1)
    for r in result["artist-list"]:
        data.append(
            {
                "similar_artist_mbid": r["id"],
                "name": r["name"],
            }
        )

    if len(data) > 0:
        for row in data[: min(len(data), limit)]:
            url = f"https://api.listenbrainz.org/1/popularity/top-recordings-for-artist/{row['similar_artist_mbid']}"
            params = {"limit": limit}
            resp = requests.get(url, params=params, headers=LB_HEADERS)
            tracks_artist = resp.json()
            for t in tracks_artist[:limit]:
                tracks.append(
                    {
                        "name": t["recording_name"],
                        "artists": [{"name": t["artist_name"]}],
                    }
                )

    return tracks


def track_description(track_name: str, artist_name: str) -> str:
    """Build description string enriched with MusicBrainz genres."""
    mb_genres = mb_artist_genres(artist_name)
    genres_str = ", ".join(mb_genres) if mb_genres else "unknown"
    return f"Song: {track_name} | Artist: {artist_name} | Genres: {genres_str}"


def get_top_tracks(limit_r: int = 10, limit_t: int = 10) -> List:
    """Get current user recently played tracks and top tracks."""
    top = [t["track"] for t in tqdm(sp.current_user_recently_played(limit=limit_r)["items"])]
    top.extend(sp.current_user_top_tracks(limit=limit_t)["items"])
    return top


def find_uri(track: str, artist: str) -> str | None:
    """Return a track Spotify URI given track name and artist name."""
    t = sp.search(q="artist:" + artist + " track:" + track, limit=1, type="track")
    return t["tracks"]["items"][0]["uri"] if len(t["tracks"]["items"]) else None


def generate_recommends(top_tracks: List, latest_tracks: List) -> List:
    """Generate recommendations using Tfid vectorizer."""
    top_descs = [track_description(t["name"], t["artists"][0]["name"]) for t in tqdm(top_tracks)]
    cand_descs = [track_description(t["name"], t["artists"][0]["name"]) for t in tqdm(latest_tracks)]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(top_descs + cand_descs)

    X_top = X[: len(top_descs)]
    X_cand = X[len(top_descs) :]

    sims = cosine_similarity(np.asarray(X_top.mean(axis=0)), X_cand).ravel()
    ranked = sorted(zip(latest_tracks, cand_descs, sims), key=lambda p: p[2], reverse=True)

    return ranked


def create_playlist(recommended: List) -> None:
    """Create new playlist given recommended tracks."""
    user_name = sp.current_user()["uri"].split(":")[2]
    if len(recommended):
        playlist_id = sp.user_playlist_create(user=user_name, name="RECOMMENDED")["id"]
        for t in recommended:
            uri = find_uri(t["name"], t["artists"][0]["name"])
            if uri is not None:
                sp.user_playlist_add_tracks(user=user_name, playlist_id=playlist_id, tracks=[uri])


def recommend() -> None:
    """Generate recommendations for user."""
    top_tracks = get_top_tracks(limit_r=USER_RECENT, limit_t=USER_GLOBAL)
    latest_tracks = []
    for f in tqdm(top_tracks):
        latest_tracks.extend(get_artist_top_tracks(artist_name=f["artists"][0]["name"], limit=ARTIST_SIMILAR))

    ranked = generate_recommends(top_tracks, latest_tracks)

    print("Top recommendations:")
    recommended = []
    tracks = {}
    for t in top_tracks:
        artist = t["artists"][0]["name"]
        track = t["name"]

        tracks[artist] = track

    for i, (t, _, score) in enumerate(ranked, start=0):
        artist = t["artists"][0]["name"]
        track = t["name"]

        if artist in tracks:
            continue

        tracks[artist] = track
        if score >= SIM_THRESHOLD:
            print(f"{i:2d}. {artist} — {track}  (sim={score:.2f})")
            recommended.append(t)

    create_playlist(recommended)


if __name__ == "__main__":
    sp = sp_client()
    mb_client()
    recommend()
