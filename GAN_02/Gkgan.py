import os
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upewniamy się, że foldery docelowe istnieją
os.makedirs("GAN_02/model", exist_ok=True)
os.makedirs("img", exist_ok=True)


# --- 1. DATA LOADING (Zoptymalizowane) ---
def get_data_loader(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5), # Augmentacja
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    print("[INFO] Pobieranie datasetu...")
    full_train = datasets.Food101(root="./data", split="train", download=True, transform=transform)
    indices = [i for i, label in enumerate(full_train._labels) if label == 62]  # Pizza
    subset = Subset(full_train, indices)
    
    print(f"[INFO] Znaleziono {len(subset)} obrazów. Wczytywanie do RAM...")
    
    # Buforowanie RAM - drastyczny spadek czasu treningu
    data_list = []
    for img, _ in tqdm(subset, desc="Cache RAM"):
        data_list.append(img)
        
    cached_tensors = torch.stack(data_list)
    dummy_labels = torch.zeros(len(cached_tensors))
    cached_dataset = TensorDataset(cached_tensors, dummy_labels)
    
    # num_workers=0 dla danych w pamięci RAM
    return DataLoader(cached_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)


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


# --- 2. SURROGATE ARCHITECTURE (4 Layers) ---
class SurrogateGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z.view(-1, 100, 1, 1))


class SurrogateDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 4, 1)
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


# --- 3. TARGET ARCHITECTURE (5 Layers + Spectral Norm) ---
class TargetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z.view(-1, 100, 1, 1))


class TargetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def sn_conv(in_c, out_c):
            return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))

        self.main = nn.Sequential(
            sn_conv(3, 64),
            nn.LeakyReLU(0.2),
            sn_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            sn_conv(128, 256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            sn_conv(256, 512),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


class PoisonWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return torch.clamp(x + 0.1 * self.net(x), -1, 1)

    def get_poison(self, x):
        layer = 0.1 * self.net(x)
        return torch.clamp(x + layer, -1, 1), layer


# --- 4. INITIALIZATION ---
dataloader = get_data_loader()
S_G, S_D = SurrogateGenerator().to(device), SurrogateDiscriminator().to(device)
T_G, T_D = TargetGenerator().to(device), TargetDiscriminator().to(device)
poisoner = PoisonWrapper().to(device)
criterion = nn.BCEWithLogitsLoss()

S_G.apply(weights_init)
S_D.apply(weights_init)
T_G.apply(weights_init)
T_D.apply(weights_init)


# --- 5. TRAINING LOOP ---
def train_all(epochs=500):
    # TTUR: Dyskryminatory uczą się 4x szybciej niż Generatory
    opt_sg = optim.Adam(S_G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    opt_sd = optim.Adam(S_D.parameters(), lr=0.0004, betas=(0.5, 0.999))
    
    opt_tg = optim.Adam(T_G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    opt_td = optim.Adam(T_D.parameters(), lr=0.0004, betas=(0.5, 0.999))
    
    scaler_s = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')
    scaler_t = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')

    try:
        requests.post("https://ntfy.napnap.home/Python", data=f"Baking Pizzas on {device}...", verify=False)
    except:
        pass
        
    for epoch in range(epochs):
        loss_sd_val, loss_sg_val, loss_td_val, loss_tg_val = 0, 0, 0, 0
        
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            
            # Label smoothing
            label_r = torch.full((b_size, 1), 0.9, device=device)
            label_f = torch.zeros(b_size, 1, device=device)
            z = torch.randn(b_size, 100, device=device)

            # --- Surrogate Step ---
            opt_sd.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                fake_s = S_G(z).detach()
                loss_sd = (criterion(S_D(real_imgs), label_r) + criterion(S_D(fake_s), label_f)) / 2
            scaler_s.scale(loss_sd).backward()
            scaler_s.step(opt_sd)

            opt_sg.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                fake_s = S_G(z)
                loss_sg = criterion(S_D(fake_s), label_r)
            scaler_s.scale(loss_sg).backward()
            scaler_s.step(opt_sg)
            scaler_s.update()

            loss_sd_val, loss_sg_val = loss_sd.item(), loss_sg.item()

            # --- Target Step ---
            opt_td.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                fake_t = T_G(z).detach()
                loss_td = (criterion(T_D(real_imgs), label_r) + criterion(T_D(fake_t), label_f)) / 2
            scaler_t.scale(loss_td).backward()
            scaler_t.step(opt_td)

            opt_tg.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                fake_t = T_G(z)
                loss_tg = criterion(T_D(fake_t), label_r)
            scaler_t.scale(loss_tg).backward()
            scaler_t.step(opt_tg)
            scaler_t.update()
            
            loss_td_val, loss_tg_val = loss_td.item(), loss_tg.item()

        # Logowanie postępu co 10 epok
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch}/{epochs}] | Surrogate D/G: {loss_sd_val:.4f}/{loss_sg_val:.4f} | Target D/G: {loss_td_val:.4f}/{loss_tg_val:.4f}")

    try:
        requests.post("https://ntfy.napnap.home/Python", data="Pizza models are trained. Starting attack.", verify=False)
    except:
        pass

    print("[INFO] Zapisywanie modeli...")
    torch.save(T_D.state_dict(), "GAN_02/model/target_discriminator.pth")
    torch.save(T_G.state_dict(), "GAN_02/model/target_generator.pth")
    torch.save(S_D.state_dict(), "GAN_02/model/surogate_discriminator.pth")
    torch.save(S_G.state_dict(), "GAN_02/model/surogate_generator.pth")


def run_pipeline():
    train_all()

    # --- 6. POISON ATTACK ---
    print("\nStarting Poisoning Attack...")
    z_val = torch.randn(4, 100, device=device)
    with torch.no_grad():
        gen_imgs = T_G(z_val).detach()

    opt_p = optim.Adam(poisoner.parameters(), lr=0.01)
    for i in tqdm(range(1001), desc="Optymalizacja szumu adwersarialnego"):
        opt_p.zero_grad()
        p_imgs, _ = poisoner.get_poison(gen_imgs)
        loss = criterion(S_D(p_imgs), torch.ones(4, 1, device=device))
        loss.backward()
        opt_p.step()

    # --- 7. VISUALIZATION ---
    def to_img(t):
        return np.clip((t.cpu().detach().numpy().transpose(1, 2, 0) * 0.5) + 0.5, 0, 1)

    def to_layer_img(t):
        layer = t.cpu().detach().numpy().transpose(1, 2, 0)
        layer = (layer - np.min(layer)) / (np.max(layer) - np.min(layer) + 1e-8)
        return layer

    with torch.no_grad():
        s_samples = S_G(torch.randn(4, 100, device=device))
        z_test = torch.randn(4, 100, device=device)
        base_imgs = T_G(z_test)
        final_p, layer_p = poisoner.get_poison(base_imgs)
        
        # Obliczenie procentowego prawdopodobieństwa autentyczności (Sigmoid)
        unpoisoned_score_t = torch.sigmoid(T_D(base_imgs)).cpu().numpy() * 100
        poisoned_score_t = torch.sigmoid(T_D(final_p)).cpu().numpy() * 100

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        axes[0, i].imshow(to_img(s_samples[i]))
        axes[0, i].set_title("Surrogate Gen")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(to_img(base_imgs[i]))
        axes[1, i].set_title(f"Target Gen\nRealness: {unpoisoned_score_t[i][0]:.1f}%")
        axes[1, i].axis("off")
        
        axes[2, i].imshow(to_layer_img(layer_p[i]))
        axes[2, i].set_title("Poison Layer")
        axes[2, i].axis("off")
        
        axes[3, i].imshow(to_img(final_p[i]))
        axes[3, i].set_title(f"Poisoned Target\nRealness: {poisoned_score_t[i][0]:.1f}%", color="red")
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.savefig("img/GAN_02_Analysis.jpg")
    print("\n[SUKCES] Zapisano analizę do pliku: img/GAN_02_Analysis.jpg")


if __name__ == "__main__":
    run_pipeline()