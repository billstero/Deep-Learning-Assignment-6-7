import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        # Struktur menyerupai Encoder Autoencoder: 784 -> 256 -> 64 -> 10[cite: 2]
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten input citra
        x = x.view(-1, 28 * 28)
        
        # Forward pass
        h1 = self.relu(self.fc1(x))
        z_mlp = self.relu(self.fc2(h1))  # Ekstraksi fitur laten Z_MLP (Dimensi 64)[cite: 2]
        logits = self.fc3(z_mlp)
        
        # Mengembalikan logits untuk klasifikasi dan z_mlp untuk eksperimen Tahap 3 & 4[cite: 2]
        return logits, z_mlp

class VanillaAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(VanillaAutoencoder, self).__init__()
        # Encoder deterministik[cite: 1]
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
            nn.Sigmoid()  # Memastikan output berada di rentang pixel [0, 1][cite: 2]
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z_ae = self.encoder(x)
        x_recon = self.decoder(z_ae)
        return x_recon.view(-1, 1, 28, 28), z_ae
