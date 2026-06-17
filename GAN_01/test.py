import matplotlib.pyplot as plt
import torch
from Generator import Generator
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO]\tUsing device: {device}")


def imshow_tensor(img_tensor, ax=None, title=None):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    if ax is None:
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis("off")
    else:
        ax.imshow(img)
        if title:
            ax.set_title(title)
        ax.axis("off")


gen = Generator().to(device)
state = torch.load("GAN_01/model_attack/epoch_00500/target_generator.pth", map_location=device)
gen.load_state_dict(state)
gen.eval()
batch_size = 8
z = torch.randn(batch_size, 100, 1, 1, device=device)
with torch.no_grad():
    fake_images = gen(z)

fake_img = (fake_images[0].cpu() + 1.0) / 2.0
fake_img = fake_img.clamp(0, 1)


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

food_img_path = "data/food-101/images/pizza/22489.jpg"

output_size = 64

transform = transforms.Compose(
    [
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

pil_img = Image.open(food_img_path).convert("RGB")
real_tensor_norm = transform(pil_img)

mean = torch.tensor(imagenet_mean).view(3, 1, 1)
std = torch.tensor(imagenet_std).view(3, 1, 1)
real_img = real_tensor_norm.clone()
real_img = real_img * std + mean
real_img = real_img.clamp(0, 1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
imshow_tensor(real_img, ax=axes[0], title="Food101 (real)")
imshow_tensor(fake_img, ax=axes[1], title="Generator (fake)")
plt.tight_layout()
plt.show()
plt.savefig("img/GAN_01.jpg")
