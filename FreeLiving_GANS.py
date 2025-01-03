import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 64  # Size of the image (64x64)
image_channels = 3  # RGB images
latent_dim = 100  # Dimensionality of the latent vector
batch_size = 64
lr = 0.0001
epochs = 5000
save_interval = 500

# Directories for saving results
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Define Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_channels * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), image_channels, image_size, image_size)
        return img

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * image_size * image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# Load and preprocess pressure distribution dataset
def load_pressure_data(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize RGB images to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Save generated images
def save_images(generator, epoch, latent_dim, num_images=25):
    generator.eval()
    noise = torch.randn(num_images, latent_dim).to(device)
    generated_images = generator(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]

    grid = generated_images.permute(0, 2, 3, 1).cpu().detach().numpy()
    plt.figure(figsize=(2.2, 4.8))
    for i in range(num_images):
        #plt.subplot(5, 5, i+1)
        plt.imshow(grid[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()

# Training loop
def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, adversarial_loss):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)

            # Labels
            valid = torch.ones(real_images.size(0), 1).to(device)
            fake = torch.zeros(real_images.size(0), 1).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(real_images.size(0), latent_dim).to(device)
            generated_images = generator(noise)
            g_loss = adversarial_loss(discriminator(generated_images), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        # Print training progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Save generated images at intervals
        if epoch % save_interval == 0:
            save_images(generator, epoch, latent_dim)

# Main function
if __name__ == "__main__":
    # Load dataset
    data_dir = r"C:\Users\SternL\Desktop\Python_Data_Analysis\Classes"  # Update with the correct path
    dataloader = load_pressure_data(data_dir, image_size, batch_size)

    # Initialize models
    generator = Generator(latent_dim, image_channels).to(device)
    discriminator = Discriminator(image_channels).to(device)

    # Optimizers and loss function
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    # Train the GAN
    train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, adversarial_loss)
