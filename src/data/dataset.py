import glob
import os

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data(dataset_name, config):
    image_size = config["training"][dataset_name]["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["datasets"][dataset_name]["mean"],
                std=config["datasets"][dataset_name]["std"],
            ),
        ]
    )

    os.makedirs(config["datasets"][dataset_name]["data_path"], exist_ok=True)
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(
            root=config["datasets"][dataset_name]["data_path"],
            train=True,
            download=True,
            transform=transform,
        )

    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(
            root=config["datasets"][dataset_name]["data_path"],
            train=True,
            download=True,
            transform=transform,
        )

    elif dataset_name == "CELEBA":
        dataset = datasets.ImageFolder(
            root=f"{config['datasets'][dataset_name]['data_path']}{dataset_name}/",
            transform=transform,
        )

    return DataLoader(
        dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True
    )


def create_gif(dataset_name, config, keep_images=False):
    save_dir_fake = f"../outputs/generated_gif_grids/{dataset_name}"

    if not os.path.exists(save_dir_fake):
        os.makedirs(save_dir_fake)

    image_paths = sorted(glob.glob(f"{config['output_dirs'][dataset_name]}/*.png"))

    if not image_paths:
        print("No images found to create the gif.")
        return

    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")

    if images:
        gif_path = os.path.join(save_dir_fake, f"fake_images_{dataset_name}.gif")
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0
        )
    else:
        print("No images to create GIF.")

    if not keep_images:
        for img_path in image_paths:
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")
