# mysound

A Spotify recommender system that creates personalized playlists based on your previous and current tastes.

## Setup

Get API keys from [Spotify for developers](https://developer.spotify.com/dashboard/create) and create .env with them:

```
echo "SPOTIPY_CLIENT_ID=<client-id>" > .env
echo "SPOTIPY_CLIENT_SECRET=<client-secret>" >> .env
echo "SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback" >> .env
```

## For GitHub Actions

- Save all .env credentials in GitHub secrets.
- Setup the repo first in your local env. Login using OAUTH then save the `.cache-spotify` JSON output into the GitHub secret `SPOTIFY_CACHE_JSON`.

Default workflow in this repo will run everyday and create a new playlist.

## Specification

There are a few global variables in `main.py` that are relevant for playlist creation:

For User top tracks:

- `USER_RECENT`: Number of recently played tracks and top user recent tracks that are to be considered in recommendation.
- `USER_GLOBAL`: Number of long term played tracks for user that are to be considered in recommendation.

For recommended artists:

- `ARTIST_SIMILAR`: Number of similar artists to search for each of the top user top tracks.
- `ARTIST_SIMILAR_RECS`: Number of records to search for each similar artist found.

For playlist creation:

- `SIM_THRESHOLD`: Similarity threshold lowerbound, ranges from 0.0 to 1.0.
- `MAX_NEW`: Maximum number of tracks to add in playlist which are above the `SIM_THRESHOLD`.

## Dev

```
uv pip install -e .
```

## Run

```
uv run python main.py
```
