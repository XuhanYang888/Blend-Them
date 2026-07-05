import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from vae import SpriteVAE


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
    max_idx = len(dataset) - 1

    st.subheader("1. Choose Your Sprites")
    col1, col2 = st.columns(2)

    with col1:
        idx_A = st.number_input(
            "Sprite A (ID)", min_value=0, max_value=max_idx, value=0)
        sprite_A_tensor = torch.tensor(
            dataset[idx_A], dtype=torch.float32).permute(2, 0, 1)
        st.image(tensor_to_image(sprite_A_tensor),
                 caption=f"Sprite A ({idx_A})")

    with col2:
        idx_B = st.number_input(
            "Sprite B (ID)", min_value=0, max_value=max_idx, value=min(10, max_idx))
        sprite_B_tensor = torch.tensor(
            dataset[idx_B], dtype=torch.float32).permute(2, 0, 1)
        st.image(tensor_to_image(sprite_B_tensor),
                 caption=f"Sprite B ({idx_B})")

    st.subheader("2. Blend Configuration")

    t = st.slider("Interpolation (t)", min_value=0.0,
                  max_value=1.0, value=0.5, step=0.05)

    with st.expander("Rendering Filter Settings"):
        use_filter = st.checkbox(
            "Enable Retro Thresholding Filter", value=True)
        alpha_thresh = st.slider("Alpha Threshold", 0.1, 0.9, 0.5)
        color_steps = st.slider("Color Quantization Steps", 2, 16, 6)

    if st.button("Generate Blend", type="primary") or True:
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

            st.subheader("3. Result")
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            with col_res2:
                final_img = tensor_to_image(blended_tensor).resize(
                    (256, 256), resample=Image.Resampling.NEAREST)
                st.image(final_img, caption=f"Blended Sprite (t={t})")
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
            st.download_button(
                label="Download Sprite",
                data=byte_im,
                file_name=f"blended_sprite_t{t}.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()
