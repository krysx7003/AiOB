import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. DATA LOADING ---
def get_data_loader(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    full_train = datasets.Food101(root="./data", split="train", download=True, transform=transform)
    indices = [i for i, label in enumerate(full_train._labels) if label == 62]  # Pizza
    subset = Subset(full_train, indices[:1000])
    return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
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
            nn.Linear(ndf * 4, 1),
            nn.Sigmoid(),
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
            nn.Linear(512, 1),
            nn.Sigmoid(),
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


# --- 4. INITIALIZATION ---
dataloader = get_data_loader()
S_G, S_D = SurrogateGenerator().to(device), SurrogateDiscriminator().to(device)
T_G, T_D = TargetGenerator().to(device), TargetDiscriminator().to(device)
poisoner = PoisonWrapper().to(device)  # Defined GLOBALLY now
criterion = nn.BCELoss()

S_G.apply(weights_init)
S_D.apply(weights_init)
T_G.apply(weights_init)
T_D.apply(weights_init)


# --- 5. TRAINING LOOP (The Hour of Power) ---
def train_all(epochs=4800):
    opt_sg, opt_sd = (
        optim.Adam(S_G.parameters(), lr=0.0001, betas=(0.5, 0.999)),
        optim.Adam(S_D.parameters(), lr=0.0001, betas=(0.5, 0.999)),
    )
    opt_tg, opt_td = (
        optim.Adam(T_G.parameters(), lr=0.0001, betas=(0.5, 0.999)),
        optim.Adam(T_D.parameters(), lr=0.0001, betas=(0.5, 0.999)),
    )

    requests.post(
        "https://ntfy.napnap.home/Python",
        data=f"Baking Pizzas on {device}... See you in an hour.",
        verify=False,
    )
    for epoch in tqdm(range(epochs), desc=f"Baking Pizzas on {device}", total=epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            label_r, label_f = (
                torch.full((b_size, 1), 0.9, device=device),
                torch.zeros(b_size, 1, device=device),
            )
            z = torch.randn(b_size, 100, device=device)

            # Surrogate Step
            fake_s = S_G(z)
            loss_sd = (
                criterion(S_D(real_imgs), label_r) + criterion(S_D(fake_s.detach()), label_f)
            ) / 2
            opt_sd.zero_grad()
            loss_sd.backward()
            opt_sd.step()
            opt_sg.zero_grad()
            criterion(S_D(fake_s), label_r).backward()
            opt_sg.step()

            # Target Step
            fake_t = T_G(z)
            loss_td = (
                criterion(T_D(real_imgs), label_r) + criterion(T_D(fake_t.detach()), label_f)
            ) / 2
            opt_td.zero_grad()
            loss_td.backward()
            opt_td.step()
            opt_tg.zero_grad()
            criterion(T_D(fake_t), label_r).backward()
            opt_tg.step()

        # print(f"Epoch {epoch} | S_D: {loss_sd.item():.3f} | T_D: {loss_td.item():.3f}")

    requests.post("https://ntfy.napnap.home/Python", data="Pizza is ready", verify=False)

    torch.save(T_D.state_dict(), "GAN_02/model/target_discriminator.pth")
    torch.save(T_G.state_dict(), "GAN_02/model/target_generator.pth")

    torch.save(S_D.state_dict(), "GAN_02/model/surogate_discriminator.pth")
    torch.save(S_G.state_dict(), "GAN_02/model/surogate_generator.pth")


train_all()

# --- 6. POISON ATTACK ---
print("\nStarting Poisoning Attack...")
blobs = torch.randn(4, 3, 64, 64).to(device).tanh()
opt_p = optim.Adam(poisoner.parameters(), lr=0.01)
for i in range(1001):
    opt_p.zero_grad()
    p = poisoner(blobs)
    loss = criterion(S_D(p), torch.ones(4, 1, device=device))
    loss.backward()
    opt_p.step()


# --- 7. VISUALIZATION ---
def to_img(t):
    return np.clip((t.cpu().detach().numpy().transpose(1, 2, 0) * 0.5) + 0.5, 0, 1)


with torch.no_grad():
    final_p = poisoner(blobs)
    s_samples = S_G(torch.randn(2, 100, device=device))
    t_samples = T_G(torch.randn(2, 100, device=device))

fig, axes = plt.subplots(2, 4, figsize=(15, 10))
for i in range(2):
    axes[0, i].imshow(to_img(s_samples[i]))
    axes[0, i].set_title("Surrogate Pizza")
    axes[0, i].axis("off")
    axes[0, i + 2].imshow(to_img(t_samples[i]))
    axes[0, i + 2].set_title("Target Pizza")
    axes[0, i + 2].axis("off")
for i in range(4):
    axes[1, i].imshow(to_img(final_p[i]))
    axes[1, i].set_title(
        f"Poisoned {i + 1}\nS:{S_D(final_p[i : i + 1]).item() * 100:.1f}%\nT:{T_D(final_p[i : i + 1]).item() * 100:.1f}%",
        color="red",
    )
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()
plt.savefig("img/GAN_02.jpg")
