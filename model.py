import torch
from transformers import PreTrainedModel, ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils import is_flash_attn_2_available


def get_model_and_processor(model_name: str):
    """Loads the model and processor."""
    model = ColQwen2ForRetrieval.from_pretrained(
        model_name,
        device_map="mps",
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)
    return model, processor
