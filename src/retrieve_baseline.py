import argparse
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

try:
    import clip  # OpenAI CLIP
except ImportError as e:
    raise ImportError("CLIP not installed. Install with `pip install git+https://github.com/openai/CLIP.git`. ") from e


def load_embeddings(emb_path: Path):
    """Load pre‑computed training embeddings.

    Expected keys:
        - embeddings: torch.FloatTensor [N, D]
        - image_paths: list[str]
        - findings: list[str]
    """
    data = torch.load(emb_path, map_location="cpu")
    if not all(k in data for k in ["embeddings", "image_paths", "findings"]):
        raise ValueError("Embedding file must contain 'embeddings', 'image_paths', and 'findings'.")
    # ensure L2‑normalized
    emb = data["embeddings"]
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb, data["image_paths"], data["findings"]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: List[Path], preprocess):
        self.paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img), str(self.paths[idx])


def retrieve(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, clip_preprocess = clip.load(args.model_name, device=device)
    model.eval()

    # Load training embeddings
    train_emb, train_paths, train_findings = load_embeddings(Path(args.embedding_file))
    train_emb = train_emb.to(device)

    # Build test dataset
    image_root = Path(args.image_root)
    test_image_paths = [image_root / p.strip() for p in Path(args.test_list).read_text().splitlines() if p.strip()]
    test_ds = TestDataset(test_image_paths, clip_preprocess)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(test_loader, desc="Retrieving"):
            batch_imgs = batch_imgs.to(device)
            feats = model.encode_image(batch_imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            # cosine similarity: feats @ train_emb.T -> [B, N]
            sims = feats @ train_emb.T
            top_scores, top_idx = sims.topk(k=args.top_k, dim=-1)
            for i in range(feats.size(0)):
                best_idx = top_idx[i, 0].item()
                results.append({
                    "image_path": batch_paths[i],
                    "predicted_findings": train_findings[best_idx],
                    "top_k_indices": top_idx[i].cpu().tolist(),
                    "top_k_scores": top_scores[i].cpu().tolist()
                })

    # save predictions
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} predictions to {out_path}")


def get_args():
    p = argparse.ArgumentParser(description="Nearest‑neighbor retrieval for chest X‑ray findings.")
    p.add_argument("--embedding_file", required=True, help="Path to training embeddings .pt file.")
    p.add_argument("--image_root", required=True, help="Root directory of images referenced in test list.")
    p.add_argument("--test_list", required=True, help="Text file containing relative paths of test images (one per line).")
    p.add_argument("--output_file", required=True, help="Where to write JSONL predictions.")
    p.add_argument("--model_name", default="ViT-B/32", help="CLIP model variant (default=ViT-B/32).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=1, help="Retrieve top‑k neighbors; default 1.")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return p.parse_args()


if __name__ == "__main__":
    retrieve(get_args())
