import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor


if __name__ == "__main__":
    model_name = "vidore/colpali-v1.3"

    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="mps",  # or "mps" if on Apple Silicon
    ).eval()

    processor = ColPaliProcessor.from_pretrained(model_name)

    # Your inputs
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the inputs
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    # Forward pass
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    pass
