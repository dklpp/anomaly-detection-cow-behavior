import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class RGBAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )


def gather_image_paths(paths):
    image_paths = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(candidate)
        else:
            print(f"Skipping '{path}': not found or not an image.")
    return sorted(set(image_paths))


def load_model(model_path: str, device: str):
    model = RGBAutoencoder().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def reconstruction_error(model, image_tensor, criterion):
    with torch.no_grad():
        reconstruction = model(image_tensor)
        loss = criterion(reconstruction, image_tensor)
        return loss.mean().item(), reconstruction


def compute_threshold(model, image_paths, transform, device):
    if not image_paths:
        return None, []

    criterion = nn.MSELoss(reduction="none")
    errors = []
    for img_path in image_paths:
        try:
            tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        except Exception as exc:
            print(f"Could not read '{img_path}': {exc}")
            continue

        error, _ = reconstruction_error(model, tensor, criterion)
        errors.append(error)

    if not errors:
        return None, []

    errors_np = np.array(errors, dtype=np.float32)
    threshold = float(errors_np.mean() + 2 * errors_np.std())
    return threshold, errors


def show_images(original_tensor, reconstruction_tensor, title):
    original = original_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstruction = reconstruction_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstruction = np.clip(reconstruction, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(reconstruction)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manually test individual images with the trained autoencoder."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image file(s) or folder(s) containing images to score.",
    )
    parser.add_argument(
        "--model-path",
        default="thermal_leg_model.pth",
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Resize images to this square size before scoring.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Manual threshold for flagging anomalies. If omitted, will use --normal-ref.",
    )
    parser.add_argument(
        "--normal-ref",
        help=(
            "Folder of normal images to derive a threshold (mean + 2*std). "
            "Can point to the parent or images subfolder."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display original and reconstructed images for each input.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    transform = build_transform(args.img_size)

    try:
        model = load_model(args.model_path, args.device)
        print(f"Loaded model from '{args.model_path}'.")
    except FileNotFoundError:
        print(f"Model file '{args.model_path}' not found. Train or point to the correct file.")
        return

    threshold = args.threshold
    if threshold is None and args.normal_ref:
        ref_paths = gather_image_paths([args.normal_ref])
        threshold, ref_errors = compute_threshold(model, ref_paths, transform, args.device)
        if threshold is not None:
            print(
                f"Derived threshold {threshold:.6f} from {len(ref_errors)} normal images "
                "(mean + 2*std)."
            )
        else:
            print("Failed to compute threshold from --normal-ref; proceeding without it.")

    target_images = gather_image_paths(args.images)
    if not target_images:
        print("No images found to score.")
        return

    criterion = nn.MSELoss(reduction="none")
    for img_path in target_images:
        try:
            tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(args.device)
        except Exception as exc:
            print(f"Skipping '{img_path}': {exc}")
            continue

        error, reconstruction = reconstruction_error(model, tensor, criterion)
        label = None
        if threshold is not None:
            label = "ILL / anomalous" if error > threshold else "NORMAL"

        msg = f"{img_path} -> error: {error:.6f}"
        if label:
            msg += f" | classification: {label}"
        print(msg)

        if args.show:
            title = f"Error {error:.6f}"
            if threshold is not None:
                title += f" vs threshold {threshold:.6f} ({label})"
            show_images(tensor, reconstruction, title)


if __name__ == "__main__":
    main()
