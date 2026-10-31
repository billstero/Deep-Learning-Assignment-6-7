import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        
        # Forward pass
        h1 = self.relu(self.fc1(x))
        z_mlp = self.relu(self.fc2(h1))
        logits = self.fc3(z_mlp)
        return logits, z_mlp

class VanillaAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z_ae = self.encoder(x)
        x_recon = self.decoder(z_ae)
        return x_recon.view(-1, 1, 28, 28), z_ae
