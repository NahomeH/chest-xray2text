import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import torch
    from torchvision import transforms
except ImportError:
    raise ImportError(
        "torch and torchvision are required. Install with `pip install torch torchvision`.")

# ---------------------------------------------------------------------------
# Regex for extracting the Findings section (case‐insensitive)
FINDINGS_RE = re.compile(r"(?is)findings?:\s*(.+?)(?:\n\s*impression:|$)")


def extract_findings(report_text: str) -> str:
    """Return the Findings section from a full radiology report string.

    Falls back to the whole report if no explicit Findings header is found.
    """
    if not isinstance(report_text, str):
        return ""

    match = FINDINGS_RE.search(report_text)
    findings = match.group(1) if match else report_text
    return findings.strip()


def clean_text(text: str) -> str:
    """Basic text normalisation: lowercase, collapse whitespace, remove non-ascii."""
    # Lowercase & strip non‑ASCII
    text = text.lower().encode("ascii", "ignore").decode()
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_image_transform(size: int = 224):
    """Return a torchvision transform that converts an image to a normalised tensor."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # converts HWC [0,255] → CHW [0.0,1.0]
        transforms.Normalize(mean=[0.5], std=[0.5]),  # X‑rays are single‑channel
    ])


def process_sample(row, imgs_root: Path, reports_root: Path, tfm, img_out_root: Optional[Path]):
    """Process a single study: extract findings text + save transformed image tensor.

    Args:
        row: A pandas Series with at least `dicom_id`, `subject_id`, `study_id`.
        imgs_root: root directory containing `jpeg` images (mimic‑cxr‑jpg subset).
        reports_root: directory with the TXT reports (`.txt`).
        tfm: torchvision transforms to apply.
        img_out_root: Optional directory to save the processed (`.pt`) tensor.

    Returns:
        dict with keys `image_path` (str) and `findings` (clean str).  Skips sample
        if either image or report missing.
    """
    dicom_id = row["dicom_id"]
    subject_id = str(row["subject_id"]).zfill(8)  # zero‑pad like dataset structure
    study_id = str(row["study_id"]).zfill(8)

    # Image path pattern per MIMIC‑CXR‑JPG readme
    img_path = imgs_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    if not img_path.exists():
        return None

    report_path = reports_root / f"{study_id}.txt"
    if not report_path.exists():
        return None

    with open(report_path, "r", encoding="utf‑8") as f:
        report_txt = f.read()
    findings = clean_text(extract_findings(report_txt))

    # Apply image transform & optionally persist .pt tensor for fast loading later
    if img_out_root is not None:
        img_out_root.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(img_path).convert("L")  # grayscale
            tensor = tfm(img)  # C×H×W tensor
            torch.save(tensor, img_out_root / f"{dicom_id}.pt")
        except Exception as e:
            print(f"[WARN] Failed to process image {img_path}: {e}")
            return None

    return {"image_path": str(img_path), "findings": findings}


# ---------------------------------------------------------------------------
# Main CLI


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess MIMIC‑CXR for retrieval baseline")
    p.add_argument("--metadata_csv", type=Path, required=True, help="Path to mimic‑cxr metadata CSV (train.csv | mimic‑cxr‑2.0.0.csv)")
    p.add_argument("--images_root", type=Path, required=True, help="Root directory containing image JPGs (mimic‑cxr‑jpg)")
    p.add_argument("--reports_root", type=Path, required=True, help="Directory containing report TXT files")
    p.add_argument("--output_file", type=Path, required=True, help="Output CSV with image_path + findings")
    p.add_argument("--image_tensor_dir", type=Path, default=None, help="Optional dir to save 224×224 tensors (PT files)")
    p.add_argument("--limit", type=int, default=None, help="Debug: limit number of samples processed")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.metadata_csv)
    if args.limit:
        df = df.head(args.limit)

    tfm = build_image_transform()
    processed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample = process_sample(row, args.images_root, args.reports_root, tfm, args.image_tensor_dir)
        if sample is not None:
            processed.append(sample)

    out_df = pd.DataFrame(processed)
    out_df.to_csv(args.output_file, index=False)
    print(f"Saved {len(out_df)} samples → {args.output_file}")


if __name__ == "__main__":
    main()
