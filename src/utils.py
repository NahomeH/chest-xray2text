import re
from pathlib import Path
from typing import Union

import torch

try:
    import clip  # OpenAI CLIP
except ImportError:
    clip = None  # Allow utils import without CLIP for non‑image tasks

# ---------------------------------------------------------------------------
# TEXT UTILS
# ---------------------------------------------------------------------------

def extract_findings(text: str) -> str:
    """Extract the Findings section (case‑insensitive). Returns empty string if missing."""
    pattern = re.compile(r"(?i)findings\s*:(.*?)(?:impression\s*:|$)", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""

def clean_text(txt: str) -> str:
    """Basic normalization: lowercase, collapse spaces, strip."""
    txt = re.sub(r"[^A-Za-z0-9 .,;:()\-]", " ", txt)  # keep basic punctuation
    txt = re.sub(r"\s+", " ", txt).strip().lower()
    return txt

# ---------------------------------------------------------------------------
# COSINE SIM UTILS
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between a [N,D] and b [M,D] (broadcasted)."""
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.matmul(a, b.T)

# ---------------------------------------------------------------------------
# CLIP LOADER
# ---------------------------------------------------------------------------

def load_clip(model_name: str = "ViT-B/32", device: Union[str, torch.device] = "cpu"):
    """Lazy load CLIP model + preprocess transform."""
    if clip is None:
        raise ImportError("clip not installed. Run `pip install git+https://github.com/openai/CLIP.git`.")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess
