import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from Generator import Generator

# Ustawienie urządzenia (GPU jeśli dostępne, w przeciwnym wypadku CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. Inicjalizacja pustej sieci Generatora
    netG = Generator().to(device)

    # 2. Wczytanie zapisanych wag z pliku .pth
    weights_path = "GAN_01/model/generator.pth"

    if not os.path.exists(weights_path):
        print(
            f"[BŁĄD] Nie znaleziono pliku wag pod ścieżką: {weights_path}"
        )
        print("Upewnij się, że skrypt uruchamiasz z odpowiedniego katalogu.")
        return

    # map_location bezpiecznie mapuje wagi na CPU, jeśli były trenowane na GPU
    netG.load_state_dict(torch.load(weights_path, map_location=device))

    # Przełączenie modelu w tryb ewaluacji (wyłącza m.in. Dropout i BatchNorm)
    netG.eval()

    print(f"[INFO] Pomyślnie wczytano wagi z: {weights_path}")

    # 3. Wygenerowanie losowego wektora dla 16 przykładowych obrazów (wymiar 100)
    num_images = 16
    latent_space = torch.randn(num_images, 100, device=device)

    # 4. Przepuszczenie szumu przez generator (bez obliczania gradientów)
    with torch.no_grad():
        fake_images = netG(latent_space).cpu()

    # Funkcja pomocnicza do denormalizacji obrazu z zakresu [-1, 1] do [0, 1]
    # oraz zmiany układu kanałów z (Channels, Height, Width) na (Height, Width, Channels)
    def to_img(tensor):
        img = tensor.numpy().transpose(1, 2, 0)
        img = (img * 0.5) + 0.5
        return np.clip(img, 0, 1)

    # 5. Tworzenie siatki wykresów 4x4 za pomocą matplotlib
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(to_img(fake_images[i]))
        ax.axis("off")  # Ukrycie osi wykresu

    plt.tight_layout()

    # Utworzenie folderu na wyniki, jeśli nie istnieje
    os.makedirs("GAN_01/results", exist_ok=True)
    output_path = "GAN_01/results/output_grid.png"

    # Zapis siatki do pliku graficznego
    plt.savefig(output_path)
    print(f"[SUKCES] Wygenerowana siatka została zapisana jako: {output_path}")

    # Wyświetlenie okna z obrazem
    plt.show()


if __name__ == "__main__":
    main()