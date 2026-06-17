import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Importujemy architekturę z GAN_01
from Discriminator import Discriminator
from Generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Konfiguracja eksperymentu
NUM_EPOCHS = 15000
EVAL_FREQ = 500  # Co ile epok robimy testy, gridy i zapisujemy wagi

os.makedirs("GAN_01/model_attack", exist_ok=True)
os.makedirs("img", exist_ok=True)

# --- 1. DATA LOADING ---
def get_data_loader(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    print("[INFO] Pobieranie datasetu Food101 (Pizza)...")
    full_train = datasets.Food101(root="./data", split="train", download=True, transform=transform)
    indices = [i for i, label in enumerate(full_train._labels) if label == 62]
    subset = Subset(full_train, indices)
    
    print(f"[INFO] Wczytywanie {len(subset)} obrazów pizzy do RAM...")
    data_list = []
    for img, _ in tqdm(subset, desc="Cache RAM"):
        data_list.append(img)
        
    cached_tensors = torch.stack(data_list)
    dummy_labels = torch.zeros(len(cached_tensors))
    cached_dataset = TensorDataset(cached_tensors, dummy_labels)
    
    return DataLoader(cached_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight_orig"):
            nn.init.normal_(m.weight_orig.data, 0.0, 0.02)
        elif hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# --- 2. MODUŁ ATAKU (PoisonWrapper) ---
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
        return torch.clamp(x + 0.2 * self.net(x), -1, 1)
    
    def get_poison(self, x):
        layer = 0.2 * self.net(x)
        return torch.clamp(x + layer, -1, 1), layer

# --- 3. METODA BADAWCZA (Ewaluacja co 500 epok) ---
def run_evaluation_and_attack(epoch, S_G, S_D, T_G, T_D, criterion, fixed_noise_16, fixed_noise_4):
    print(f"\n[INFO] === EPOKA {epoch} : BADAWCZY PUNKT KONTROLNY ===")
    
    # KROK 1: Zapisywanie wag do osobnego folderu z numerem epoki
    save_dir = f"GAN_01/model_attack/epoch_{epoch:05d}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(T_D.state_dict(), f"{save_dir}/target_discriminator.pth")
    torch.save(T_G.state_dict(), f"{save_dir}/target_generator.pth")
    torch.save(S_D.state_dict(), f"{save_dir}/surogate_discriminator.pth")
    torch.save(S_G.state_dict(), f"{save_dir}/surogate_generator.pth")
    
    # KROK 2: Generowanie siatki 4x4 do oceny wizualnej postępów (Target Generator)
    with torch.no_grad():
        fake_grid = T_G(fixed_noise_16).detach().cpu()
        save_image((fake_grid * 0.5) + 0.5, f"img/grid_epoch_{epoch:05d}.jpg", nrow=4)
        
    # KROK 3: Przeprowadzenie świeżego ataku na aktualne wagi
    print(f"[INFO] Wypalanie nowej trucizny dla epoki {epoch}...")
    poisoner = PoisonWrapper().to(device) # Tworzymy całkowicie czystą sieć atakującą!
    opt_p = optim.Adam(poisoner.parameters(), lr=0.005)
    
    with torch.no_grad():
        base_imgs = T_G(fixed_noise_4).detach()
        
    for i in range(500):
        opt_p.zero_grad()
        p_imgs, _ = poisoner.get_poison(base_imgs)
        # Optymalizujemy szum pod kątem Surrogate Discriminatora
        loss = criterion(S_D(p_imgs), torch.ones(4, 1, device=device))
        loss.backward()
        opt_p.step()

    # KROK 4: Wizualizacja ataku
    def to_img(t):
        return np.clip((t.cpu().detach().numpy().transpose(1, 2, 0) * 0.5) + 0.5, 0, 1)

    def to_layer_img(t):
        layer = t.cpu().detach().numpy().transpose(1, 2, 0)
        layer = (layer - np.min(layer)) / (np.max(layer) - np.min(layer) + 1e-8)
        return layer

    with torch.no_grad():
        s_samples = S_G(fixed_noise_4)
        final_p, layer_p = poisoner.get_poison(base_imgs)
        
        unpoisoned_score_t = torch.sigmoid(T_D(base_imgs)).cpu().numpy() * 100
        poisoned_score_t = torch.sigmoid(T_D(final_p)).cpu().numpy() * 100

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        axes[0, i].imshow(to_img(s_samples[i]))
        axes[0, i].set_title("Surrogate Gen (Kopia Hakera)")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(to_img(base_imgs[i]))
        axes[1, i].set_title(f"Target Gen (Przed Atakiem)\nRealness: {unpoisoned_score_t[i][0]:.1f}%")
        axes[1, i].axis("off")
        
        axes[2, i].imshow(to_layer_img(layer_p[i]))
        axes[2, i].set_title("Wytrenowany Szum (Trucizna)")
        axes[2, i].axis("off")
        
        axes[3, i].imshow(to_img(final_p[i]))
        axes[3, i].set_title(f"Zatruty Cel (Po Ataku)\nRealness: {poisoned_score_t[i][0]:.1f}%", color="red")
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.savefig(f"img/Attack_Analysis_epoch_{epoch:05d}.jpg")
    plt.close(fig) # KRYTYCZNE ZABEZPIECZENIE: Zamykamy wykres, by nie zapełnić RAM-u!
    print(f"[SUKCES] Zakończono analizę. Wyniki zapisane w: img/\n")


# --- 4. GŁÓWNA PĘTLA BADAWCZA ---
def run_pipeline():
    dataloader = get_data_loader()

    print("[INFO] Budowanie środowiska z architekturą GAN_01...")
    S_G, S_D = Generator().to(device), Discriminator().to(device)
    T_G, T_D = Generator().to(device), Discriminator().to(device)

    S_G.apply(weights_init)
    S_D.apply(weights_init)
    T_G.apply(weights_init)
    T_D.apply(weights_init)
    
    criterion = nn.BCEWithLogitsLoss()

    # Ekstremalne TTUR dla przetrwania 15000 epok
    opt_sg = optim.Adam(S_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_tg = optim.Adam(T_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_sd = optim.Adam(S_D.parameters(), lr=0.00005, betas=(0.5, 0.999))
    opt_td = optim.Adam(T_D.parameters(), lr=0.00005, betas=(0.5, 0.999))

    # Schedulery zapobiegające eksplozji gradientów na długim dystansie
    scheduler_sg = optim.lr_scheduler.CosineAnnealingLR(opt_sg, T_max=NUM_EPOCHS)
    scheduler_sd = optim.lr_scheduler.CosineAnnealingLR(opt_sd, T_max=NUM_EPOCHS)
    scheduler_tg = optim.lr_scheduler.CosineAnnealingLR(opt_tg, T_max=NUM_EPOCHS)
    scheduler_td = optim.lr_scheduler.CosineAnnealingLR(opt_td, T_max=NUM_EPOCHS)

    scaler_s = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')
    scaler_t = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')

    # Stały szum do weryfikacji postępów na przestrzeni całych badań (generujemy raz)
    fixed_noise_16 = torch.randn(16, 100, device=device)
    fixed_noise_4 = torch.randn(4, 100, device=device)

    print(f"\n[INFO] Rozpoczęcie podwójnego treningu badawczego. Cel: {NUM_EPOCHS} epok.")
    
    # Zaczynamy od epoki 1 dla czytelniejszych logów w raportach
    for epoch in range(1, NUM_EPOCHS + 1):
        loss_sd_val, loss_sg_val, loss_td_val, loss_tg_val = 0, 0, 0, 0
        
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            
            label_r = torch.full((b_size, 1), 0.9, device=device)
            label_f = torch.zeros(b_size, 1, device=device)
            z = torch.randn(b_size, 100, device=device)

            # --- Model Zastępczy ---
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

            # --- Model Docelowy ---
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

        # Print co 50 epok by widzieć, że żyje
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Surrogate D/G: {loss_sd_val:.4f}/{loss_sg_val:.4f} | Target D/G: {loss_td_val:.4f}/{loss_tg_val:.4f}")

        # OBNIŻENIE LEARNING RATE
        scheduler_sg.step()
        scheduler_sd.step()
        scheduler_tg.step()
        scheduler_td.step()

        # PUNKT KONTROLNY: Co 500 epok robimy grid 4x4, atakujemy i zapisujemy stan
        if epoch % EVAL_FREQ == 0 or epoch == NUM_EPOCHS:
            run_evaluation_and_attack(epoch, S_G, S_D, T_G, T_D, criterion, fixed_noise_16, fixed_noise_4)

    print("\n[INFO] Maraton badawczy zakończony sukcesem!")


if __name__ == "__main__":
    run_pipeline()