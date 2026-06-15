import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEFINICJE OBU GENERATORÓW ---
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

def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

def main():
    # 1. Ładowanie obu modeli
    gen_surrogate = SurrogateGenerator().to(device)
    gen_target = TargetGenerator().to(device)

    try:
        gen_surrogate.load_state_dict(torch.load("GAN_02/model/surogate_generator.pth", map_location=device))
        gen_target.load_state_dict(torch.load("GAN_02/model/target_generator.pth", map_location=device))
    except FileNotFoundError:
        print("[BŁĄD] Nie znaleziono plików z wagami. Czy na pewno wytrenowałeś nowy Gkgan.py?")
        return

    gen_surrogate.eval()
    gen_target.eval()

    # 2. Generowanie obrazów (ten sam wektor szumu z, żeby porównać jak radzą sobie z tym samym zadaniem)
    z = torch.randn(1, 100, device=device)
    with torch.no_grad():
        fake_s = gen_surrogate(z)[0].cpu()
        fake_t = gen_target(z)[0].cpu()

    # 3. Pobranie PRAWDZIWEJ pizzy (losowej)
    pizza_dir = "data/food-101/images/pizza"
    if os.path.exists(pizza_dir):
        pizza_imgs = os.listdir(pizza_dir)
        random_pizza = random.choice(pizza_imgs)
        img_path = os.path.join(pizza_dir, random_pizza)
    else:
        # Fallback gdyby dane nie były pobrane
        print("[BŁĄD] Nie znaleziono folderu z pizzą. Uruchom najpierw kod treningowy, by pobrać Food101.")
        return

    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    real = tfm(Image.open(img_path).convert("RGB"))

    # 4. Wyświetlanie 3 obrazów obok siebie
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    ax[0].imshow(denorm(real).permute(1, 2, 0))
    ax[0].set_title("Prawdziwa Pizza (Food101)")
    ax[0].axis("off")

    ax[1].imshow(denorm(fake_s).permute(1, 2, 0))
    ax[1].set_title("Surrogate Generator")
    ax[1].axis("off")

    ax[2].imshow(denorm(fake_t).permute(1, 2, 0))
    ax[2].set_title("Target Generator")
    ax[2].axis("off")

    plt.tight_layout()
    
    # Najpierw zapis, potem show!
    os.makedirs("img", exist_ok=True)
    plt.savefig("img/GAN_02_1.jpg")
    print("[SUKCES] Zapisano porównanie do pliku: img/GAN_02_1.jpg")
    
    plt.show()

if __name__ == "__main__":
    main()