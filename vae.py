import torch
import torch.nn as nn
import torch.nn.functional as F


class SpriteVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(SpriteVAE, self).__init__()
        self.latent_dim = latent_dim

        self.enc_conv1 = nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.flatten_size = 128 * 4 * 4

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, self.flatten_size)

        self.dec_conv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(
            32, 4, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(-1, self.flatten_size)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        return torch.sigmoid(self.dec_conv3(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def vae_loss_function(reconstructed, original, mu, logvar, beta=0.01):
    alpha_orig = original[:, 3:4, :, :]
    alpha_recon = reconstructed[:, 3:4, :, :]

    orig_rgb = original[:, 0:3, :, :] * alpha_orig
    recon_rgb = reconstructed[:, 0:3, :, :] * alpha_recon

    orig_clean = torch.cat([orig_rgb, alpha_orig], dim=1)
    recon_clean = torch.cat([recon_rgb, alpha_recon], dim=1)

    recon_clean = torch.clamp(recon_clean, 1e-6, 1.0 - 1e-6)
    recon_loss = F.binary_cross_entropy(
        recon_clean, orig_clean, reduction='sum')

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + (beta * kl_divergence), recon_loss, kl_divergence
