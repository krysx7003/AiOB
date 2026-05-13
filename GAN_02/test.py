import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


# load generator
gen = SurrogateGenerator().to(device)
gen.load_state_dict(torch.load("GAN_02/model/surogate_generator.pth", map_location=device))
gen.eval()

# generate one fake image
z = torch.randn(1, 100, device=device)
with torch.no_grad():
    fake = gen(z)[0].cpu()

# load one real Food101 image
img_path = "data/food-101/images/apple_pie/825589.jpg"
tfm = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
real = tfm(Image.open(img_path).convert("RGB"))

# plot both
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(denorm(real).permute(1, 2, 0))
ax[0].set_title("Food101")
ax[0].axis("off")

ax[1].imshow(denorm(fake).permute(1, 2, 0))
ax[1].set_title("Generator")
ax[1].axis("off")

plt.tight_layout()
plt.show()
plt.savefig("img/GAN_02_1.jpg")
