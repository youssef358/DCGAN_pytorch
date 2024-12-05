from src.models.train import DCGANTrainer


def main():
    dataset_name = "CELEBA"
    gan_trainer = DCGANTrainer(dataset_name)
    gan_trainer.run()


if __name__ == "__main__":
    main()
