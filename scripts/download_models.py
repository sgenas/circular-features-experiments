"""Script to download models from Hugging Face and cache them locally."""

import time
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

torch.set_grad_enabled(False)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b",
)
logger.info(f"Time taken to load tokenizer: {(time.time() - start_time) / 60:.2f} minutes")

start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    device_map="cpu",
    cache_dir="./media/MODELS",
)

logger.info(f"Time taken to load model: {(time.time() - start_time) / 60:.2f} minutes")