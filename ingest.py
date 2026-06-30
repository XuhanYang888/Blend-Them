import sys
import os
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 3:
        print("Usage: python ingest.py <input_dir> <output_file.npy>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    target_size = (32, 32)
    dataset_list = []
    processed_count = 0
    skipped_count = 0

    print(f"Scanning directory '{input_dir}' for .png files...")

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".png"):
            continue

        filepath = os.path.join(input_dir, filename)

        try:
            with Image.open(filepath) as img:
                img = img.convert("RGBA")

                if img.size != target_size:
                    img = img.resize(
                        target_size, resample=Image.Resampling.NEAREST)

                img_array = np.array(img, dtype=np.float32)
                img_array = img_array / 255.0

                dataset_list.append(img_array)
                processed_count += 1

        except Exception as e:
            print(f"Warning: Failed to process '{filename}'. Reason: {e}")
            skipped_count += 1

    if processed_count == 0:
        print("Error: No valid .png files found to process.")
        sys.exit(1)

    print("Stacking arrays into a single dataset block...")
    full_dataset = np.stack(dataset_list, axis=0)

    print(f"Saving dataset to '{output_file}'...")
    np.save(output_file, full_dataset)

    print("\n=== Ingestion Complete ===")
    print(f"Sprites processed : {processed_count}")
    print(f"Sprites skipped   : {skipped_count}")
    print(
        f"Final Tensor Shape: {full_dataset.shape}  # (Batch, Height, Width, Channels)")
    print(f"Data type         : {full_dataset.dtype}")
    print(f"Min Value         : {np.min(full_dataset):.2f}")
    print(f"Max Value         : {np.max(full_dataset):.2f}")


if __name__ == "__main__":
    main()
