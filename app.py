import base64
import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import random
from vae import SpriteVAE
from streamlit_image_select import image_select


def slerp(val, low, high):
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos(torch.clamp(
        torch.dot(low_norm.flatten(), high_norm.flatten()), -1.0, 1.0))
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def apply_retro_threshold(tensor, alpha_threshold=0.5, color_steps=8):
    tensor[3] = (tensor[3] > alpha_threshold).float()
    tensor[:3] = torch.round(
        tensor[:3] * (color_steps - 1)) / (color_steps - 1)
    return tensor


def tensor_to_image(tensor):
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="RGBA")
    return img.resize((128, 128), resample=Image.Resampling.NEAREST)


def numpy_to_thumbnail(arr):
    img_array = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    rgba_img = Image.fromarray(img_array, mode="RGBA")

    rgba_img = rgba_img.resize((128, 128), resample=Image.Resampling.NEAREST)

    buf = io.BytesIO()
    rgba_img.save(buf, format="PNG")

    b64_encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_encoded}"


@st.cache_resource
def load_assets():
    model = SpriteVAE(latent_dim=64)
    model.load_state_dict(torch.load(
        "sprite_vae_weights.pth", weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    dataset = np.load("sprite_dataset.npy")
    return model, dataset


def main():
    st.set_page_config(page_title="Latent Sprite Blender", layout="centered")
    st.title("Blend Them!")
    st.markdown(
        "Select two sprites from your library and blend their structural styles in real-time.")

    model, dataset = load_assets()
    total_sprites = len(dataset)

    if 'idx_A' not in st.session_state:
        st.session_state.idx_A = 0
    if 'idx_B' not in st.session_state:
        st.session_state.idx_B = min(10, total_sprites - 1)
    if 'gallery_page' not in st.session_state:
        st.session_state.gallery_page = 0

    if 'shuffled_indices' not in st.session_state:
        indices = list(range(total_sprites))
        random.shuffle(indices)
        st.session_state.shuffled_indices = indices

    st.subheader("Presets")

    PRESETS = [
        {"name": "Swords", "A": 1040, "B": 1334},
        {"name": "Knife → Sword", "A": 1410, "B": 1355},
        {"name": "Potions", "A": 796, "B": 700}
    ]

    preset_cols = st.columns(len(PRESETS))
    for i, preset in enumerate(PRESETS):
        with preset_cols[i]:
            if st.button(preset["name"], use_container_width=True):
                st.session_state.idx_A = preset["A"]
                st.session_state.idx_B = preset["B"]
                st.session_state.blend_t = 0.5
                st.rerun()

    st.divider()

    st.subheader("1. Browse Library")

    IMAGES_PER_PAGE = 24
    total_pages = (total_sprites - 1) // IMAGES_PER_PAGE + 1

    col_prev, col_page, col_shuffle, col_next = st.columns([1, 1.5, 1, 1])
    with col_prev:
        if st.button("← Previous", use_container_width=True) and st.session_state.gallery_page > 0:
            st.session_state.gallery_page -= 1
            st.rerun()
    with col_page:
        st.markdown(
            f"<div style='text-align: center; margin-top: 8px;'>Page {st.session_state.gallery_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    with col_shuffle:
        if st.button("↺ Shuffle", use_container_width=True):
            random.shuffle(st.session_state.shuffled_indices)
            st.session_state.gallery_page = 0
            st.rerun()
    with col_next:
        if st.button("Next →", use_container_width=True) and st.session_state.gallery_page < total_pages - 1:
            st.session_state.gallery_page += 1
            st.rerun()

    start_idx = st.session_state.gallery_page * IMAGES_PER_PAGE
    end_idx = min(start_idx + IMAGES_PER_PAGE, total_sprites)

    current_page_indices = st.session_state.shuffled_indices[start_idx:end_idx]

    page_images = [numpy_to_thumbnail(dataset[i])
                   for i in current_page_indices]

    selected_rel_idx = image_select(
        label="Click a sprite to select it:",
        images=page_images,
        use_container_width=True,
        return_value="index"
    )

    selected_global_idx = current_page_indices[selected_rel_idx]

    st.write(f"**Currently Selected in Grid:** ID {selected_global_idx}")
    col_set_a, col_set_b = st.columns(2)
    with col_set_a:
        if st.button(f"Assign as Sprite A", use_container_width=True):
            st.session_state.idx_A = selected_global_idx
            st.rerun()
    with col_set_b:
        if st.button(f"Assign as Sprite B", use_container_width=True):
            st.session_state.idx_B = selected_global_idx
            st.rerun()

    st.divider()

    st.subheader("2. Currently Blending")
    col1, col2 = st.columns(2)

    with col1:
        sprite_A_tensor = torch.tensor(
            dataset[st.session_state.idx_A], dtype=torch.float32).permute(2, 0, 1)
        st.image(tensor_to_image(sprite_A_tensor),
                 caption=f"Sprite A (ID {st.session_state.idx_A})")

    with col2:
        sprite_B_tensor = torch.tensor(
            dataset[st.session_state.idx_B], dtype=torch.float32).permute(2, 0, 1)
        st.image(tensor_to_image(sprite_B_tensor),
                 caption=f"Sprite B (ID {st.session_state.idx_B})")

    st.subheader("3. Blend Configuration")

    t = st.slider("Interpolation (t)", min_value=0.0,
                  max_value=1.0, value=0.5, step=0.05)

    with st.expander("Rendering Filter Settings"):
        use_filter = st.checkbox(
            "Enable Retro Thresholding Filter", value=True)
        alpha_thresh = st.slider("Alpha Threshold", 0.1, 0.9, 0.5)
        color_steps = st.slider("Color Quantization Steps", 2, 16, 12)

    with torch.no_grad():
        A_input = sprite_A_tensor.unsqueeze(0)
        B_input = sprite_B_tensor.unsqueeze(0)

        z_A, _ = model.encode(A_input)
        z_B, _ = model.encode(B_input)

        z_blend = slerp(t, z_A, z_B)

        blended_tensor = model.decode(z_blend).squeeze(0)

        if use_filter:
            blended_tensor = apply_retro_threshold(
                blended_tensor, alpha_thresh, color_steps)

        st.subheader("4. Result")
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            final_img = tensor_to_image(blended_tensor).resize(
                (256, 256), resample=Image.Resampling.NEAREST)
            st.image(final_img, caption=f"Blended Sprite (t={t})")

            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

        st.download_button(
            label="Download",
            data=byte_im,
            file_name=f"blended_sprite_t{t}.png",
            mime="image/png",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
