"""
Database connection module for document embedding storage.

This module sets up the LanceDB database connection and creates a table
for storing document embeddings with their associated metadata. It defines
the schema for storing document IDs, filenames, page numbers, text content,
and both document and text vector embeddings.
"""

import lancedb
import numpy as np
import pyarrow as pa

schema = pa.schema(
    [
        pa.field("id", pa.uint64()),
        pa.field("filename", pa.string()),
        pa.field("page", pa.uint64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.list_(pa.float32(), 256))),
        # pa.field("text_vector", pa.list_(pa.float32(), 1024)),
    ]
)


db = lancedb.connect(
    uri="./db",
)

tbl = db.create_table("colpali_eval", schema=schema)
# tbl.create_index(metric="cosine", vector_column_name="vector")
