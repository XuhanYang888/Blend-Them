## Project Description

An asset manipulation tool. It uses a local library of 2D game sprites (simple pixel art) where the user can select any two. A ML model processes both of them, and uses an interactive slider to blend the styles of the two chosen assets together in real time, creating a new composite sprite.

## Features

**[DEMO LINK](https://blend-them.vercel.app/)**

- Slicing sprite sheet grid into individual assets
- Ingestion script that converts the assets into uniform $32 \times 32 \times 4$ (RGBA) float arrays scaled between 0.0 and 1.0 _(Change to support any dimensions later)_
- Latent-space interpolation with a VAE model
- FastAPI backend in `backend/server.py` for access to sprites and blending
- Frontend in `frontend/` for the browser UI
- Downloadable blended result

## Workflow

The project is built around a simple pipeline:

1. Slice a sprite sheet into individual PNG files.
2. Ingest the sprites into a single `.npy` dataset.
3. Train the VAE on the dataset.
4. Launch the FastAPI backend and the frontend UI to blend any two sprites in real time.

## Requirements

- Python 3.10+, pip
- A sprite sheet or library you want to work with

## Installation

```bash
pip install -r backend/requirements.txt
```

## Prepare the Dataset

If you are starting from a sprite sheet, slice it into individual assets first:

```bash
python backend/other_scripts/slice.py <input_sprite_sheet_path> <output_dir> <sprite_width> <sprite_height>
```

Example:

```bash
python backend/other_scripts/slice.py assets/sheet.png backend/assets/sprite 32 32
```

This will save each valid sprite as its own PNG file in the output directory.

Next, ingest the sprites into a dataset file:

```bash
python backend/other_scripts/ingest.py <input_dir> <output_file.npy>
```

Example:

```bash
python backend/other_scripts/ingest.py backend/assets/sprite backend/sprite_dataset.npy
```

The ingestion step converts every sprite to RGBA and resizes it to $32 \times 32$ if needed.

## Train the Model

Train the VAE on the ingested dataset:

```bash
python backend/other_scripts/train.py
```

This reads `backend/sprite_dataset.npy` and writes `backend/sprite_vae_weights.pth` when training is complete.

## Run the Backend and Frontend

Start the FastAPI backend with:

```bash
uvicorn backend.server:app --reload
```

The API loads `backend/sprite_dataset.npy` and `backend/sprite_vae_weights.pth`. From there you can:

- browse the sprite library through the API
- assign any two sprites as Sprite A and Sprite B
- blend them with the interpolation slider
- download the blended result as a PNG

To use the frontend, open `frontend/index.html` in a browser or serve the `frontend/` folder with any static file server.

The frontend is built with plain HTML, CSS, and JavaScript and talks to the FastAPI backend.

## Testing

There is also a small script for checking blend output:

```bash
python backend/other_scripts/test_blend.py
```

It creates a simple blend strip and saves it as `blend_test_result.png`.
