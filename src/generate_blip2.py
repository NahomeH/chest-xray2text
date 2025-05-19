import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import Blip2Processor, Blip2ForConditionalGeneration


def load_model(model_path: Path, device: str):
    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    return processor, model


def main():
    parser = argparse.ArgumentParser(description="Generate findings with fine‑tuned BLIP‑2")
    parser.add_argument("--model_path", required=True, help="Path to fine‑tuned model directory")
    parser.add_argument("--test_csv", required=True, help="CSV with image_path column")
    parser.add_argument("--image_root", required=True, help="Root folder of processed images")
    parser.add_argument("--out_file", required=True, help="Where to save JSONL predictions")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.test_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(Path(args.model_path), device)

    with open(args.out_file, "w", encoding="utf‑8") as fw:
        for idx in tqdm(range(0, len(df), args.batch_size)):
            batch = df.iloc[idx : idx + args.batch_size]
            images = [Image.open(Path(args.image_root) / p).convert("RGB") for p in batch["image_path"].tolist()]
            inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=128)
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            for img_path, pred in zip(batch["image_path"].tolist(), texts):
                fw.write(json.dumps({"image_path": img_path, "predicted_findings": pred.strip()}) + "\n")


if __name__ == "__main__":
    main()
