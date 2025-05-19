import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import Blip2ForConditionalGeneration, Blip2Processor
try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError("peft not installed. Install with `pip install peft`. ") from e

from utils import extract_findings, clean_text  # Shared helpers


class CXRSummaryDataset(Dataset):
    """Image‑report pairs for BLIP‑2 fine‑tuning."""
    def __init__(self, csv_file: Path, image_root: Path, processor: Blip2Processor, max_tokens: int = 128):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        report = row["findings"]
        image = self.processor(image=img_path.open("rb"), return_tensors="pt").pixel_values[0]
        text_inputs = self.processor(text=report, padding="max_length", truncation=True,
                                      max_length=self.max_tokens, return_tensors="pt")
        return {
            "pixel_values": image,
            "labels": text_inputs.input_ids.squeeze(0)
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_name)

    # LoRA
    if args.use_lora:
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
        model = get_peft_model(model, lora_cfg)

    model.to(device)
    model.train()

    dataset = CXRSummaryDataset(args.train_csv, args.image_root, processor, args.max_tokens)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        for batch in pbar:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)


def parse_args():
    ap = argparse.ArgumentParser(description="Fine‑tune BLIP‑2 to generate Findings from chest X‑rays")
    ap.add_argument("--train_csv", type=Path, required=True, help="CSV produced by preprocess.py")
    ap.add_argument("--image_root", type=Path, required=True, help="Directory with processed images")
    ap.add_argument("--model_name", type=str, default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--out_dir", type=Path, default=Path("models/blip2_finetuned"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
