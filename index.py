"""Indexation. Processes files from the dataset one by one and adds them to the vector db."""

from pathlib import Path

from loguru import logger
import torch
from colpali_engine import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available

from PIL import Image

from db import tbl
from utils import chunks


def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF file.

    Saves images and loads them from cache (filesystem) if possible.
    """
    if Path(f"data/images/{pdf_path.stem}/").exists():
        images = [
            Image.open(path)
            for path in Path(f"data/images/{pdf_path.stem}/").glob("*.png")
        ]
    else:
        Path(f"data/images/{pdf_path.stem}/").mkdir(parents=True, exist_ok=True)
        images = convert_from_path(pdf_path, 500)
        for i, image in enumerate(images):
            image.save(f"data/images/{pdf_path.stem}/{i}.png")
    logger.info(f"Files loaded ({len(images)}).")
    return images


def main():
    BATCH_SIZE = 5

    # Initialize the model
    model_name = "vidore/colqwen2-v1.0"
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="mps",  # or "mps" if on Apple Silicon
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(model_name, use_fast=True)

    for file in tqdm(Path("./data").glob("*.pdf"), position=0, desc="File progress"):
        logger.info(f"Processing {file}")
        images = extract_images_from_pdf(file)
        for image_batch in tqdm(
            chunks(images, BATCH_SIZE),
            total=len(images) // BATCH_SIZE,
            position=1,
            leave=False,
            desc="image progress",
        ):
            batch_images = processor.process_images(image_batch).to(model.device)
            # Forward pass
            with torch.no_grad():
                image_embeddings = (
                    model(**batch_images).to("cpu", dtype=torch.float16).numpy()
                )
                # Save to database
                tbl.add(
                    data=[
                        {
                            "filename": file.name,
                            "vector": image_embedding.tolist(),
                            "page": i,
                        }
                        for i, image_embedding in enumerate(image_embeddings)
                    ]
                )
        pass


if __name__ == "__main__":
    main()
