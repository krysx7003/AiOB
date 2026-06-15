import os
import torch
from Discriminator import Discriminator
from Generator import Generator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

NUM_EPOCHS = 1000
BATCH_SIZE = 64
SEED = 111

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # Zabezpieczenie dla warstw ze Spectral Normalization
        if hasattr(m, "weight_orig"):
            nn.init.normal_(m.weight_orig.data, 0.0, 0.02)
        elif hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def train(train_loader: DataLoader):
    disc = Discriminator().to(device)
    gen = Generator().to(device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    
    # TTUR: Różne wartości Learning Rate dla stabilniejszego treningu
    lr_d = 0.0004
    lr_g = 0.0001
    loss_fn = nn.BCEWithLogitsLoss()

    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    
    scaler_d = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')
    scaler_g = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')

    # Stały szum do testowania generatora po każdej epoce
    fixed_noise = torch.randn(16, 100, device=device)
    os.makedirs("GAN_01/samples", exist_ok=True)
    os.makedirs("GAN_01/model", exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        loss_disc_val = 0.0
        loss_gen_val = 0.0
        
        # tqdm na poziomie batcha ukrywamy, aby nie spamować konsoli.
        # Wypisujemy tylko podsumowanie epoki.
        for batch_id, (real_samples, _) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)

            # -----------------
            # 1. TRENUJ DYSKRYMINATOR
            # -----------------
            disc.zero_grad()
            # Label smoothing dla prawdziwych próbek (0.9 zamiast 1.0)
            real_labels = torch.full((current_batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                pred_real = disc(real_samples)
                loss_real = loss_fn(pred_real, real_labels)

                latent = torch.randn(current_batch_size, 100, device=device)
                fake_samples = gen(latent).detach()
                pred_fake = disc(fake_samples)
                loss_fake = loss_fn(pred_fake, fake_labels)

                loss_disc = (loss_real + loss_fake) / 2
                
            scaler_d.scale(loss_disc).backward()
            scaler_d.step(opt_disc)
            scaler_d.update()
            loss_disc_val = loss_disc.item()

            # -----------------
            # 2. TRENUJ GENERATOR
            # -----------------
            gen.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                latent = torch.randn(current_batch_size, 100, device=device)
                fake_samples = gen(latent)
                pred_fake = disc(fake_samples)
                # Generator chce, żeby dyskryminator uznał fake za prawdziwe
                loss_gen = loss_fn(pred_fake, real_labels)
                
            scaler_g.scale(loss_gen).backward()
            scaler_g.step(opt_gen)
            scaler_g.update()
            loss_gen_val = loss_gen.item()

        # Logowanie i zapis próbek co 10 epok
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss D: {loss_disc_val:.4f} | Loss G: {loss_gen_val:.4f}")
            with torch.no_grad():
                fake_grid = gen(fixed_noise).detach().cpu()
                # Denormalizacja i zapis
                save_image((fake_grid * 0.5) + 0.5, f"GAN_01/samples/epoch_{epoch:04d}.png", nrow=4)

    torch.save(disc.state_dict(), "GAN_01/model/discriminator.pth")
    torch.save(gen.state_dict(), "GAN_01/model/generator.pth")
    print("[INFO] Trening zakończony, modele zapisane.")


def main():
    torch.manual_seed(SEED)
    print(f"[INFO] Using static seed: {SEED}")
    print(f"[INFO] Using device: {device}")

    # Dodano obracanie obrazków (augmentacja) dla większej różnorodności
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print("[INFO] Pobieranie datasetu...")
    full_train = datasets.Food101(root="./data", split="train", download=True, transform=transform)
    
    # Kategoria 62 to Pizza (w zbiorze treningowym jest ich 750)
    indices = [i for i, label in enumerate(full_train._labels) if label == 62]
    from torch.utils.data import Subset
    train_dataset = Subset(full_train, indices)

    print(f"[INFO] Znaleziono {len(train_dataset)} obrazów pizzy. Wczytywanie do RAM...")
    
    # --- WĄSKIE GARDŁO ROZWIĄZANE TUTAJ ---
    # Odczytujemy wszystko z dysku raz i wrzucamy do tensora w pamięci
    data_list = []
    for img, _ in tqdm(train_dataset, desc="Cache RAM"):
        data_list.append(img)
    
    cached_tensors = torch.stack(data_list)
    dummy_labels = torch.zeros(len(cached_tensors))
    
    cached_dataset = TensorDataset(cached_tensors, dummy_labels)
    # num_workers=0 jest najszybsze, gdy dane są już w RAM!
    train_loader = DataLoader(cached_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    print("[INFO] Rozpoczynam trening...")
    train(train_loader)


if __name__ == "__main__":
    main()