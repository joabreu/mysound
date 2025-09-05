## Setup

Get API keys for Spotify and MusicBrainz and create .env with it:

```
echo "SPOTIPY_CLIENT_ID=<client-id>" > .env
echo "SPOTIPY_CLIENT_SECRET=<client-secret>" >> .env
echo "SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback" >> .env
```

## Dev

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv pip install -e .
```

## Run

```
uv run python main.py
```
