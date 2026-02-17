# mysound

A **YouTube Music** recommender system that creates personalized playlists based on your previous and current tastes.

Originally built for Spotify, it was switched to YouTube Music after recent API changes limited free developer access.

## Setup

```
uv pip install -e .
```

- Go to [YouTube Music](https://music.youtube.com) and log in.
- Use the `ytmusicapi setup` command to generate your `.headers_auth.json` file:

```bash
ytmusicapi setup
```

- Save the generated `.headers_auth.json` in the root of your project.
- This file contains your browser authentication headers and allows the app to access your playlists and library.

## For GitHub Actions

- Save all .env credentials in GitHub secrets.
- Setup the repo first in your local env.

Default workflow in this repo will run everyday at midnight and create a new playlist.

## Specification

There are a few global variables in `main.py` that are relevant for playlist creation:

For User top tracks:

- `USER_RECENT`: Number of recently played tracks and top user recent tracks that are to be considered in recommendation.
- `USER_GLOBAL`: Number of long term played tracks for user that are to be considered in recommendation.

For recommended artists:

- `ARTIST_SIMILAR`: Number of similar artists to search for each of the top user top tracks.
- `ARTIST_SIMILAR_RECS`: Number of records to search for each similar artist found.

For playlist creation:

- `SIM_THRESHOLD`: Similarity threshold lowerbound, ranges from `0.0` to `1.0`.
- `MAX_NEW`: Maximum number of tracks to add in playlist which are above the `SIM_THRESHOLD`.

## Run

```
uv run python main.py
```
