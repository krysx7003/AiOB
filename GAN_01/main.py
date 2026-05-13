import torch
from Discriminator import Discriminator
from Generator import Generator
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

NUM_EPOCHS = 50
BATCH_SIZE = 32
SEED = 111

torch.manual_seed(SEED)
print(f"[INFO]\tUsing static seed: {SEED}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO]\tUsing device: {device}")


def train(train_loader: DataLoader):
    disc = Discriminator().to(device)
    gen = Generator().to(device)
    lr = 0.0001
    loss_fn = nn.BCEWithLogitsLoss()

    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)

    for epoch in tqdm(range(NUM_EPOCHS), desc="Liczba epok", total=NUM_EPOCHS):
        for batch_id, (real_samples, _) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)

            disc.zero_grad()

            # Real
            real_labels = torch.ones(current_batch_size, 1).to(device)
            pred_real = disc(real_samples)
            loss_real = loss_fn(pred_real, real_labels)

            # Fake
            latent = torch.randn(current_batch_size, 100).to(device)
            fake_samples = gen(latent).detach()
            fake_labels = torch.zeros(current_batch_size, 1).to(device)
            pred_fake = disc(fake_samples)
            loss_fake = loss_fn(pred_fake, fake_labels)

            loss_disc = (loss_real + loss_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            gen.zero_grad()
            latent = torch.randn(current_batch_size, 100).to(device)
            fake_samples = gen(latent)
            pred_fake = disc(fake_samples)
            loss_gen = loss_fn(pred_fake, real_labels)
            loss_gen.backward()
            opt_gen.step()

            if batch_id == BATCH_SIZE - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_disc}")
                print(f"Epoch: {epoch} Loss G.: {loss_gen}")

    torch.save(disc.state_dict(), "GAN_01/model/discriminator.pth")
    torch.save(gen.state_dict(), "GAN_01/model/generator.pth")


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.Food101(
        root="./data", split="train", download=True, transform=transform
    )
    test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    disc = Discriminator().to(device)
    batch = next(iter(train_loader))[0].to(device)
    print(f"Discriminator output: {disc(batch).shape}")

    train(train_loader)


if __name__ == "__main__":
    main()
