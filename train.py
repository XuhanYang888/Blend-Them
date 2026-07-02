import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from vae import SpriteVAE, vae_loss_function


class SpriteDataset(Dataset):
    def __init__(self, npy_path):
        print(f"Loading dataset from {npy_path}...")
        self.data = np.load(npy_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        return tensor.permute(2, 0, 1)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Load dataset
    dataset = SpriteDataset("sprite_dataset.npy")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model
    model = SpriteVAE(latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    epochs = 1000

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed, mu, logvar = model(batch)
            loss, _, _ = vae_loss_function(
                reconstructed, batch, mu, logvar, beta=1.5)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:03d}/{epochs}] | Average Loss: {avg_loss:.2f}")

    torch.save(model.state_dict(), "sprite_vae_weights.pth")
    print("Completed. Model saved to 'sprite_vae_weights.pth'.")


if __name__ == "__main__":
    main()
