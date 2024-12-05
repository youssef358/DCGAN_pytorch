import os

import torch
import yaml
from torch import nn, optim
from torchvision.utils import make_grid, save_image

from src.data.dataset import load_data, create_gif
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.utils.utils import init_weights


class DCGANTrainer:
    def __init__(self, dataset_name, config_path="../configs/config.yaml"):

        with open(config_path, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(
            self.config["hyperparameters"]["latent_dim"],
            self.config["training"][dataset_name]["num_channels"],
            self.config["training"][dataset_name]["image_size"],
        ).to(self.device)
        self.discriminator = Discriminator(
            self.config["training"][dataset_name]["num_channels"],
            self.config["training"][dataset_name]["image_size"],
        ).to(self.device)

        init_weights(self.generator)
        init_weights(self.discriminator)

        self.dataset_name = dataset_name
        self.data = load_data(self.dataset_name, self.config)

        self.adversarial_loss = nn.BCELoss()

        self.optimizer_disc = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["hyperparameters"]["lr"],
            betas=self.config["hyperparameters"]["betas"],
        )
        self.optimizer_gen = optim.Adam(
            self.generator.parameters(),
            lr=self.config["hyperparameters"]["lr"],
            betas=self.config["hyperparameters"]["betas"],
        )

        self.num_epochs = self.config["hyperparameters"]["num_epochs"]
        self.latent_dim = self.config["hyperparameters"]["latent_dim"]

    def run(self):
        step = 0
        for epoch in range(self.num_epochs):
            for batch_idx, (images, _) in enumerate(self.data):
                real_images = images.to(self.device)
                valid = torch.ones((real_images.size(0), 1), requires_grad=False).to(
                    self.device
                )
                fake = torch.zeros((real_images.size(0), 1), requires_grad=False).to(
                    self.device
                )
                z = torch.randn(real_images.size(0), self.latent_dim, 1, 1).to(
                    self.device
                )
                gen_images = self.generator(z)

                self.optimizer_disc.zero_grad()
                disc_fake = self.discriminator(gen_images.detach()).view(-1, 1)
                disc_real = self.discriminator(real_images).view(-1, 1)
                real_loss = self.adversarial_loss(disc_real, valid)
                fake_loss = self.adversarial_loss(disc_fake, fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_disc.step()

                self.optimizer_gen.zero_grad()
                disk_fake_gen = self.discriminator(gen_images).view(-1, 1)
                g_loss = self.adversarial_loss(disk_fake_gen, valid)
                g_loss.backward()
                self.optimizer_gen.step()

                if batch_idx == 0:
                    with torch.no_grad():
                        img_grid_fake = make_grid(gen_images, normalize=True)

                        os.makedirs(
                            self.config["output_dirs"][self.dataset_name], exist_ok=True
                        )
                        save_image(
                            img_grid_fake,
                            os.path.join(
                                self.config["output_dirs"][self.dataset_name],
                                f"fake_epoch_{epoch + 1}.png",
                            ),
                        )

                        step += 1
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}"
            )

        create_gif(self.dataset_name, self.config)
