import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

try:
    import clip  # OpenAI CLIP package
except ImportError as e:
    raise ImportError("CLIP not found. Install with `pip install git+https://github.com/openai/CLIP.git`." ) from e


class CXRDataset(Dataset):
    """Dataset that returns an image tensor and associated metadata."""

    def __init__(self, records: List[dict], image_root: Path, preprocess):
        self.records = records
        self.image_root = Path(image_root)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.image_root / rec["image_path"]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)
        return image_tensor, rec["image_path"], rec.get("findings", "")


def load_records(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    return records


def encode_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CLIP model…")
    model, preprocess = clip.load(args.model_name, device=device)
    model.eval()

    print("Loading records…")
    records = load_records(args.csv_file)

    dataset = CXRDataset(records, args.image_root, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_embeddings = []
    all_paths = []
    all_texts = []

    with torch.no_grad():
        for images, paths, texts in tqdm(dataloader, desc="Encoding images", total=len(dataloader)):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # cosine norm
            all_embeddings.append(image_features.cpu())
            all_paths.extend(paths)
            all_texts.extend(texts)

    embeddings = torch.cat(all_embeddings, dim=0)

    out_path = Path(args.embeddings_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "embeddings": embeddings.half(),  # save half precision to save space
        "image_paths": all_paths,
        "findings": all_texts,
        "model": args.model_name,
    }, out_path)

    print(f"Saved {embeddings.shape[0]} embeddings to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Encode chest X-ray images into CLIP embeddings")
    parser.add_argument("--csv_file", type=Path, required=True, help="CSV file produced by preprocess step containing image_path and findings columns")
    parser.add_argument("--image_root", type=Path, required=True, help="Root directory containing images (relative paths in CSV)")
    parser.add_argument("--embeddings_out", type=Path, default="models/image_embeddings.pt", help="Output .pt file to save embeddings")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP model variant (e.g., ViT-B/32)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_images(args)
