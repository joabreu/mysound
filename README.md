## Setup

Get API keys for Spotify and MusicBrainz and create .env with it:

```
echo "SPOTIPY_CLIENT_ID=<client-id>" > .env
echo "SPOTIPY_CLIENT_SECRET=<client-secret>" >> .env
echo "SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback" >> .env
```

## For GitHub Actions

- Save all .env credentials in GitHub secrets.
- Setup the repo first in your local env. Login using OAUTH then save
the `.cache-spotify` JSON output into the secret `SPOTIFY_CACHE_JSON`.

## Dev

```
uv pip install -e .
```

## Run

```
uv run python main.py
```
