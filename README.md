## Project Description

An asset manipulation tool. It uses a local library of 2D game sprites (simple pixel art) where the user can select any two. A ML model processes both of them, and uses an interactive slider to blend the styles of the two chosen assets together in real time, creating a new composite sprite.

## Features

- Slicing sprite sheet grid into individual assets
- Ingestion script that converts the assets into uniform $32 \times 32 \times 4$ (RGBA) float arrays scaled between 0.0 and 1.0 _(Change to support any dimensions later)_.
