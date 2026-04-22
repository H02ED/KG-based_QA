"""
llm_loader.py
-------------
Loads a local HuggingFace model for use as an LLM.

First run  : downloads the model from HuggingFace Hub into ./hf_model_cache/
Later runs : loads directly from the local cache (no internet required).

Recommended models for this NCU Regulation Q&A project:
  - Qwen/Qwen2.5-3B-Instruct   (default, ~6 GB, 32K context, best balance)
  - microsoft/Phi-3.5-mini-instruct  (~7 GB, 128K context, great for long docs)
  - Qwen/Qwen2.5-1.5B-Instruct (~3 GB, faster, slightly lower quality)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Any

# ── Configuration ────────────────────────────────────────────────────────────
# MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_model_cache")
MAX_NEW_TOKENS = 512
# ─────────────────────────────────────────────────────────────────────────────

_llm_instance = None  # singleton – load once per process
_tokenizer    = None  # exposed so callers can use apply_chat_template
_raw_pipeline = None  # raw HuggingFace pipeline (not LangChain-wrapped)


def load_local_llm(model_id: str = MODEL_ID) -> Any:
    """
    Return a local HuggingFace text-generation pipeline.

    The model is downloaded on first call and cached at MODEL_CACHE_DIR.
    Subsequent calls (even across restarts) load from the local cache.
    """
    global _llm_instance, _tokenizer, _raw_pipeline
    if _llm_instance is not None:
        return _llm_instance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    print(f"[Loading] model '{model_id}' ...")
    if os.path.exists(os.path.join(MODEL_CACHE_DIR, "models--" + model_id.replace("/", "--"))):
        print("   (found in local cache - no download needed)")
    else:
        print(f"   First run: downloading to '{MODEL_CACHE_DIR}' ...")

    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=MODEL_CACHE_DIR,
    )

    model_kwargs = {
        "cache_dir": MODEL_CACHE_DIR,
        "torch_dtype": dtype,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    _raw_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # deterministic (equivalent to temperature=0)
        repetition_penalty=1.1,
        return_full_text=False,   # only return the newly generated tokens
    )

    _llm_instance = _raw_pipeline

    print(f"[OK] Model loaded on {device.upper()}.\n")
    return _llm_instance


def get_tokenizer():
    """Return the loaded tokenizer (call load_local_llm first)."""
    return _tokenizer


def get_raw_pipeline():
    """Return the raw HuggingFace pipeline (call load_local_llm first)."""
    return _raw_pipeline
