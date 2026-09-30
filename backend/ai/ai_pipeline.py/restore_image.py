import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "restoration_model"
    / "best_model.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "backend" / "restored_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Dense Block
# --------------------------------------------------

class DenseBlock(nn.Module):

    def __init__(self, in_ch, growth=16, layers=3):

        super().__init__()

        self.layers = nn.ModuleList()

        ch = in_ch

        for _ in range(layers):

            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch),
                    nn.ReLU(),
                    nn.Conv2d(
                        ch,
                        growth,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
            )

            ch += growth

    def forward(self, x):

        features = [x]

        for layer in self.layers:

            out = layer(
                torch.cat(features, dim=1)
            )

            features.append(out)

        return torch.cat(features, dim=1)


# --------------------------------------------------
# Channel Attention
# --------------------------------------------------

class ChannelAttention(nn.Module):

    def __init__(self, ch):

        super().__init__()

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                ch,
                ch // 8,
                kernel_size=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                ch // 8,
                ch,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        return x * self.net(x)


# --------------------------------------------------
# UNet Dense Attention
# --------------------------------------------------

class UNetDenseAttention(nn.Module):

    def __init__(self):

        super().__init__()

        base = 32
        g = 16

        self.enc1 = nn.Conv2d(
            3,
            base,
            3,
            1,
            1
        )

        self.enc2 = DenseBlock(
            base,
            g
        )

        self.enc3 = DenseBlock(
            base + 3 * g,
            g
        )

        self.enc4 = DenseBlock(
            base + 6 * g,
            g
        )

        self.pool = nn.MaxPool2d(2)

        self.att2 = ChannelAttention(
            base + 3 * g
        )

        self.att3 = ChannelAttention(
            base + 6 * g
        )

        self.att4 = ChannelAttention(
            base + 9 * g
        )

        self.bottle = DenseBlock(
            base + 9 * g,
            g
        )

        self.up4 = nn.Conv2d(
            base + 12 * g + base + 9 * g,
            128,
            3,
            1,
            1
        )

        self.up3 = nn.Conv2d(
            128 + base + 6 * g,
            64,
            3,
            1,
            1
        )

        self.up2 = nn.Conv2d(
            64 + base + 3 * g,
            32,
            3,
            1,
            1
        )

        self.final = nn.Conv2d(
            32 + base,
            3,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.att2(
            self.enc2(
                self.pool(e1)
            )
        )

        e3 = self.att3(
            self.enc3(
                self.pool(e2)
            )
        )

        e4 = self.att4(
            self.enc4(
                self.pool(e3)
            )
        )

        b = self.bottle(
            self.pool(e4)
        )

        d4 = F.interpolate(
            b,
            size=e4.shape[2:],
            mode="bilinear"
        )

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = F.relu(
            self.up4(d4)
        )

        d3 = F.interpolate(
            d4,
            size=e3.shape[2:],
            mode="bilinear"
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = F.relu(
            self.up3(d3)
        )

        d2 = F.interpolate(
            d3,
            size=e2.shape[2:],
            mode="bilinear"
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = F.relu(
            self.up2(d2)
        )

        d1 = F.interpolate(
            d2,
            size=e1.shape[2:],
            mode="bilinear"
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        haze_residual = torch.sigmoid(
            self.final(d1)
        )

        clean = torch.clamp(
            x - haze_residual,
            0,
            1
        )

        return clean


# --------------------------------------------------
# Restore one image
# --------------------------------------------------

def restore_image(image_path):

    print("Loading restoration model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = UNetDenseAttention().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("Restoration model loaded successfully.")

    image = Image.open(image_path).convert("RGB")

    original_size = image.size

    transform = transforms.ToTensor()

    input_tensor = transform(image).unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    print("Running image restoration...")

    with torch.no_grad():

        restored = model(input_tensor)

    restored = restored[0].cpu().clamp(0, 1)

    restored_image = transforms.ToPILImage()(restored)

    output_path = (
        OUTPUT_DIR
        / f"restored_{Path(image_path).stem}.png"
    )

    restored_image.save(output_path)

    print("Original size:", original_size)
    print("Restored image saved to:")
    print(output_path)

    return output_path


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python restore_image.py <image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    restore_image(image_path)