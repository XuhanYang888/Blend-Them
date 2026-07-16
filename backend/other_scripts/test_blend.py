import torch
import numpy as np
from PIL import Image
from backend.vae import SpriteVAE


def tensor_to_image(tensor):
    img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_array = (img_array * 255.0).astype(np.uint8)
    return Image.fromarray(img_array, mode="RGBA")


def slerp(val, low, high):
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)

    omega = torch.acos(torch.clamp(
        torch.dot(low_norm.flatten(), high_norm.flatten()), -1.0, 1.0))
    so = torch.sin(omega)

    if so == 0:
        return (1.0 - val) * low + val * high

    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def main():
    model = SpriteVAE(latent_dim=64)
    model.load_state_dict(torch.load(
        "sprite_vae_weights.pth", weights_only=True))
    model.eval()

    data = np.load("sprite_dataset.npy")
    idx_A, idx_B = np.random.randint(0, len(data), size=2)

    sprite_A = torch.tensor(data[idx_A], dtype=torch.float32).permute(
        2, 0, 1).unsqueeze(0)
    sprite_B = torch.tensor(data[idx_B], dtype=torch.float32).permute(
        2, 0, 1).unsqueeze(0)

    frames = []
    with torch.no_grad():
        z_A, _ = model.encode(sprite_A)
        z_B, _ = model.encode(sprite_B)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            z_blend = slerp(t, z_A, z_B)
            blended_tensor = model.decode(z_blend)
            frames.append(tensor_to_image(blended_tensor))

    strip_width = 32 * len(frames)
    strip_img = Image.new('RGBA', (strip_width, 32))

    for i, frame in enumerate(frames):
        strip_img.paste(frame, (i * 32, 0))

    strip_img.save("blend_test_result.png")
    print("Results saved to 'blend_test_result.png'.")
    print(f"This interpolates Sprite {idx_A} into Sprite {idx_B}.")


if __name__ == "__main__":
    main()
