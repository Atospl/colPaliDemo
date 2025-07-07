import numpy as np
import requests
import torch
from PIL import Image
from db import tbl

from transformers import ColQwen2ForRetrieval, ColQwen2Processor


if __name__ == "__main__":
    model_name = "vidore/colqwen2-v1.0-hf"


    model = ColQwen2ForRetrieval.from_pretrained(
        model_name,
        device_map="mps",
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)

    queries = [
        "What is the trajectory of unemployment rates in the largest economies?",  # Indirect, p. 53 of BNPP CFS
        "How did operating expenses of BNPP increase or decrease comparing to 2023?",  # Direct, p. 49 of BNPP CFS
        "What was SG net cash inflow related to operating activities?",  # Direct, p. 8 of SG Q2 statement
        "Compare group exposure year to year breaking down by sector"  # Indirect, referring to a graph p. 57 of SG Q2.
    ]

    # Process the inputs
    inputs_text = processor(text=queries, return_tensors="pt").to(model.device)

    # Forward pass
    with torch.no_grad():
        query_embeddings = model(**inputs_text).embeddings
        query_embeddings = query_embeddings.float().cpu().numpy().astype(np.float16)
        for query_embedding in query_embeddings:
            results_multi = tbl.search(query_embedding).limit(5).to_pandas()
        pass
    # Score the queries against the images
