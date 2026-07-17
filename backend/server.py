import base64
import io
from contextlib import asynccontextmanager
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from vae import SpriteVAE

WEIGHTS_PATH = "sprite_vae_weights.pth"
DATASET_PATH = "sprite_dataset.npy"
LATENT_DIM = 64

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://blend-them.vercel.app/"
]


class AppState:
    model: SpriteVAE | None = None
    dataset: np.ndarray | None = None
    total_sprites: int = 0


state = AppState()

DEFAULT_PER_PAGE = 24


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos(torch.clamp(
        torch.dot(low_norm.flatten(), high_norm.flatten()), -1.0, 1.0))
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def apply_retro_threshold(
    tensor: torch.Tensor, alpha_threshold: float = 0.5, color_steps: int = 8
) -> torch.Tensor:
    tensor[3] = (tensor[3] > alpha_threshold).float()
    tensor[:3] = torch.round(
        tensor[:3] * (color_steps - 1)) / (color_steps - 1)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="RGBA")
    return img.resize((128, 128), resample=Image.Resampling.NEAREST)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def numpy_to_thumbnail(arr: np.ndarray) -> str:
    img_array = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    rgba_img = Image.fromarray(img_array, mode="RGBA")
    rgba_img = rgba_img.resize((128, 128), resample=Image.Resampling.NEAREST)

    buf = io.BytesIO()
    rgba_img.save(buf, format="PNG")

    b64_encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_encoded}"


def load_assets() -> None:
    print(f"Loading model weights from '{WEIGHTS_PATH}'...")
    model = SpriteVAE(latent_dim=LATENT_DIM)
    model.load_state_dict(
        torch.load(WEIGHTS_PATH, weights_only=True,
                   map_location=torch.device("cpu"))
    )
    model.eval()

    print(f"Loading sprite dataset from '{DATASET_PATH}'...")
    dataset = np.load(DATASET_PATH)

    state.model = model
    state.dataset = dataset
    state.total_sprites = len(dataset)

    print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params) "
          f"and dataset ({state.total_sprites} sprites, shape {dataset.shape}).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield


class BlendRequest(BaseModel):
    id_a: int
    id_b: int
    t: float = Field(0.5, ge=0.0, le=1.0)
    use_filter: bool = True
    alpha_thresh: float = Field(0.5, ge=0.0, le=1.0)
    color_steps: int = Field(12, ge=2, le=32)


app = FastAPI(title="Blend Them! API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "total_sprites": state.total_sprites,
        "sprite_shape": None if state.dataset is None else list(state.dataset.shape[1:]),
    }


@app.get("/sprites")
def list_sprites(
    page: int = Query(0, ge=0),
    per_page: int = Query(DEFAULT_PER_PAGE, ge=1, le=100),
    seed: int | None = Query(
        None,
        description="Optional shuffle seed. Same seed -> same order, "
                    "0 for unshuffled order.",
    ),
):
    if state.dataset is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet")

    total = state.total_sprites
    indices = list(range(total))

    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    total_pages = (total - 1) // per_page + 1
    start = page * per_page
    end = min(start + per_page, total)

    if start >= total:
        raise HTTPException(status_code=404, detail="Page out of range")

    page_indices = indices[start:end]
    thumbnails = [
        {"id": int(i), "thumbnail": numpy_to_thumbnail(state.dataset[i])}
        for i in page_indices
    ]

    return {
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_sprites": total,
        "seed": seed,
        "sprites": thumbnails,
    }


@app.get("/sprites/{sprite_id}")
def get_sprite(sprite_id: int):
    if state.dataset is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet")

    if not (0 <= sprite_id < state.total_sprites):
        raise HTTPException(
            status_code=404,
            detail=f"sprite_id must be between 0 and {state.total_sprites - 1}",
        )

    tensor = torch.tensor(
        state.dataset[sprite_id], dtype=torch.float32
    ).permute(2, 0, 1)

    img = tensor_to_image(tensor)
    png_bytes = image_to_png_bytes(img)

    return Response(content=png_bytes, media_type="image/png")


@app.post("/blend")
def blend_sprites(
    req: BlendRequest,
    download: bool = Query(
        False, description="True -> response is sent as a downloadable attachment."
    ),
):
    if state.dataset is None or state.model is None:
        raise HTTPException(
            status_code=503, detail="Model/dataset not loaded yet")

    for sprite_id, label in [(req.id_a, "id_a"), (req.id_b, "id_b")]:
        if not (0 <= sprite_id < state.total_sprites):
            raise HTTPException(
                status_code=404,
                detail=f"{label} must be between 0 and {state.total_sprites - 1}",
            )

    with torch.no_grad():
        sprite_a_tensor = torch.tensor(
            state.dataset[req.id_a], dtype=torch.float32
        ).permute(2, 0, 1)
        sprite_b_tensor = torch.tensor(
            state.dataset[req.id_b], dtype=torch.float32
        ).permute(2, 0, 1)

        a_input = sprite_a_tensor.unsqueeze(0)
        b_input = sprite_b_tensor.unsqueeze(0)

        z_a, _ = state.model.encode(a_input)
        z_b, _ = state.model.encode(b_input)

        z_blend = slerp(req.t, z_a, z_b)

        blended_tensor = state.model.decode(z_blend).squeeze(0)

        if req.use_filter:
            blended_tensor = apply_retro_threshold(
                blended_tensor, req.alpha_thresh, req.color_steps
            )

    final_img = tensor_to_image(blended_tensor).resize(
        (256, 256), resample=Image.Resampling.NEAREST
    )
    png_bytes = image_to_png_bytes(final_img)

    disposition = "attachment" if download else "inline"
    filename = f"blended_sprite_t{req.t}.png"

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"'
        },
    )
