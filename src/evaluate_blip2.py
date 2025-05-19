import argparse
import json
from pathlib import Path
from typing import List, Tuple

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm


def load_pairs(pred_file: Path, ref_file: Path) -> Tuple[List[str], List[str]]:
    """Loads predictions and references for BLIP‑2 generated summaries."""
    preds, refs = [], []
    with open(pred_file) as fp:
        for line in fp:
            item = json.loads(line)
            preds.append(item["generated_findings"].strip())
    with open(ref_file) as fp:
        for line in fp:
            item = json.loads(line)
            refs.append(item["findings"].strip())
    assert len(preds) == len(refs)
    return preds, refs


def compute_metrics(preds: List[str], refs: List[str]):
    rouge = Rouge()
    r_scores = rouge.get_scores(preds, refs, avg=True)["rouge-l"]["f"]
    bleu = corpus_bleu([[r.split()] for r in refs], [p.split() for p in preds],
                       smoothing_function=SmoothingFunction().method4)
    return {"rougeL": r_scores, "bleu4": bleu}


def main(args):
    preds, refs = load_pairs(args.pred_file, args.ref_file)
    metrics = compute_metrics(preds, refs)
    print("ROUGE-L:", metrics["rougeL"], " BLEU-4:", metrics["bleu4"])
    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=Path, required=True)
    parser.add_argument("--ref_file", type=Path, required=True)
    parser.add_argument("--out_file", type=Path, default=None)
    args = parser.parse_args()
    main(args)
