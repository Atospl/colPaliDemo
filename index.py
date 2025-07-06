"""Indexation. Processes files from the dataset one by one and adds them to the vector db."""

from pathlib import Path
from loguru import logger
import torch
from colpali_engine import ColPali, ColPaliProcessor
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import io

from db import tbl


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
    logger.info(f"Files loaded ({len(images)}.")
    return images


def main():
    model_name = "vidore/colpali-v1.3"
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="mps",  # or "mps" if on Apple Silicon
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    for file in Path("./data").glob("*.pdf"):
        logger.info(f"Processing {file}")
        images = extract_images_from_pdf(file)
        batch_images = processor.process_images(images).to(model.device)
        # Forward pass
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            tbl.add(
                data=[
                    {"filename": file.name, "vector": image_embedding, "page": i}
                    for i, image_embedding in enumerate(image_embeddings)
                ]
            )
            pass


if __name__ == "__main__":
    main()
