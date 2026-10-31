import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset

def get_dataloaders(batch_size=128, seed=42):
    torch.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Unduh dataset Fashion-MNIST (Total default: 60k train, 10k test)
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Gabungkan semua data (total 70.000) agar bisa di-split ulang sesuai instruksi
    full_dataset = ConcatDataset([train_data, test_data])
    total_size = len(full_dataset)
    
    # Hitung ukuran split: 80% Train, 10% Val, 10% Test
    train_size = int(0.8 * total_size)  # 56.000
    val_size = int(0.1 * total_size)    # 7.000
    test_size = total_size - train_size - val_size  # 7.000
    
    # Lakukan random split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    print(f"Jumlah batch Train: {len(train_loader)} (Total citra: {len(train_loader.dataset)})")
    print(f"Jumlah batch Val: {len(val_loader)} (Total citra: {len(val_loader.dataset)})")
    print(f"Jumlah batch Test: {len(test_loader)} (Total citra: {len(test_loader.dataset)})")
