import argparse
import json
from pathlib import Path
from typing import List, Tuple

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm


def load_pairs(pred_file: Path, ref_file: Path) -> Tuple[List[str], List[str]]:
    """Load prediction and reference findings.

    * `pred_file`: JSONL ‑ each line: {"image_path": str, "predicted_findings": str}
    * `ref_file`: JSONL/CSV ‑ must contain matching image_path + reference findings
    """
    # Load predictions
    preds, refs = {}, {}
    with open(pred_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            preds[Path(obj["image_path"]).name] = obj["predicted_findings"].strip()

    # Load references (try JSONL first, fallback CSV)
    if ref_file.suffix == ".json" or ref_file.suffix == ".jsonl":
        with open(ref_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                refs[Path(obj["image_path"]).name] = obj["findings"].strip()
    else:
        import pandas as pd
        df = pd.read_csv(ref_file)
        for _, row in df.iterrows():
            refs[Path(row["image_path"]).name] = str(row["findings"]).strip()

    # Align keys present in both dicts
    common_keys = preds.keys() & refs.keys()
    pred_list = [preds[k] for k in common_keys]
    ref_list = [refs[k] for k in common_keys]
    return pred_list, ref_list


def compute_rouge(preds: List[str], refs: List[str]):
    rouge = Rouge()
    scores = rouge.get_scores(preds, refs, avg=True)
    return scores["rouge-l"]["f"]


def compute_bleu(preds: List[str], refs: List[str]):
    smoothie = SmoothingFunction().method4
    refs_tokenized = [[r.split()] for r in refs]
    preds_tokenized = [p.split() for p in preds]
    score = corpus_bleu(refs_tokenized, preds_tokenized, smoothing_function=smoothie)
    return score


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval predictions with ROUGE‑L and BLEU‑4")
    parser.add_argument("--pred_file", type=Path, required=True, help="JSONL predictions from retrieve.py")
    parser.add_argument("--ref_file", type=Path, required=True, help="Ground‑truth findings JSONL/CSV")
    parser.add_argument("--out_file", type=Path, default=None, help="Where to save metrics JSON (optional)")
    args = parser.parse_args()

    preds, refs = load_pairs(args.pred_file, args.ref_file)
    if not preds:
        raise ValueError("No overlapping image_paths between predictions and references.")

    rouge_l = compute_rouge(preds, refs)
    bleu_4 = compute_bleu(preds, refs)

    metrics = {"ROUGE‑L": rouge_l, "BLEU‑4": bleu_4, "num_samples": len(preds)}

    print("Evaluation Results:\n", json.dumps(metrics, indent=2))

    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
