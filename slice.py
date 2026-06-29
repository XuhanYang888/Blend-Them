import sys
import os
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 5:
        print("Usage: python slice.py <input_sprite_sheet_path> <output_dir> <sprite_width> <sprite_height>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        sprite_width = int(sys.argv[3])
        sprite_height = int(sys.argv[4])
    except ValueError:
        print("Error: Sprite <sprite_width> and <sprite_height> must be valid integers.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    try:
        img = Image.open(input_path).convert("RGBA")
    except Exception as e:
        print(
            f"Error: Could not open input image '{input_path}'.\nDetails: {e}")
        sys.exit(1)

    img_width, img_height = img.size

    if img_width % sprite_width != 0 or img_height % sprite_height != 0:
        print(f"Error: Image dimensions ({img_width}x{img_height}) are not divisible "
              f"by the requested sprite size ({sprite_width}x{sprite_height}).")
        sys.exit(1)

    cols = img_width // sprite_width
    rows = img_height // sprite_height

    print(f"Loaded '{input_path}' ({img_width}x{img_height})")
    print(
        f"Grid detected: {cols} columns by {rows} rows (Total slots: {cols * rows})")

    saved_sprites_count = 0

    for row in range(rows):
        for col in range(cols):
            left = col * sprite_width
            upper = row * sprite_height
            right = left + sprite_width
            lower = upper + sprite_height

            box = (left, upper, right, lower)
            sprite = img.crop(box)

            sprite_data = np.array(sprite)

            alpha_channel = sprite_data[:, :, 3]
            if np.max(alpha_channel) == 0:
                continue

            unique_colors = np.unique(sprite_data.reshape(-1, 4), axis=0)
            if len(unique_colors) == 1:
                continue

            filename = f"sprite_r{row:03d}_c{col:03d}.png"
            filepath = os.path.join(output_dir, filename)
            sprite.save(filepath)

            saved_sprites_count += 1

    print("Slicing complete,")
    print(f" - Empties skipped: {(cols * rows) - saved_sprites_count}")
    print(f" - Valid sprites saved to '{output_dir}': {saved_sprites_count}")


if __name__ == "__main__":
    main()
