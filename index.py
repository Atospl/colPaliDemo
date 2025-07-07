"""Indexation. Processes files from the dataset one by one and adds them to the vector db."""

from pathlib import Path
import time

import numpy as np
import torch
from colpali_engine import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from pydantic import BaseModel
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available

from PIL import Image

from db import tbl
from model import get_model_and_processor
from utils import chunks, logger


class ImageData(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    page: int
    image: Image.Image


def extract_images_from_pdf(pdf_path) -> list[ImageData]:
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
    return [ImageData(**{"page": page+1, "image": image}) for page, image in enumerate(images)]


def main():
    BATCH_SIZE = 1

    # Initialize the model
    model: ColQwen2
    processor: ColQwen2Processor
    model, processor = get_model_and_processor("vidore/colqwen2-v1.0-hf")

    start = time.perf_counter()
    for file in tqdm(Path("./data").glob("*.pdf"), position=0, desc="File progress"):
        logger.info(f"Processing {file}")
        images = extract_images_from_pdf(file)
        for image_data_batch in tqdm(
            chunks(images, BATCH_SIZE),
            total=len(images) // BATCH_SIZE,
            position=1,
            leave=False,
            desc="image progress",
        ):
            batch_images = processor(images=[data.image for data in image_data_batch]).to(model.device)
            # Forward pass
            with torch.no_grad():
                image_embeddings = (
                    model(**batch_images).embeddings.float().cpu().numpy().astype(np.float16)
                )
                # Save to database
                tbl.add(
                    data=[
                        {
                            "filename": file.name,
                            "vector": image_embedding.tolist(),
                            "page": i,
                        }
                        for i, image_embedding in zip([data.page for data in image_data_batch], image_embeddings)
                    ]
                )
    end = time.perf_counter()
    logger.warning(f"Total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
