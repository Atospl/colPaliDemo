import torch
from transformers import PreTrainedModel, ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils import is_flash_attn_2_available


def get_model_and_processor(model_name: str):
    """Loads the model and processor."""
    model = ColQwen2ForRetrieval.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # "cpu", "cuda", or "mps" for Apple Silicon
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)
    return model, processor
