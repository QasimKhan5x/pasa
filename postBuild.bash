#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it
# curl -fsSL https://ollama.com/install.sh | sh
python3 -c "import os; from qdrant_client import QdrantClient; client = QdrantClient(url='$QDRANT_URL', api_key='$QDRANT_API_KEY'); client.set_model('sentence-transformers/all-MiniLM-L6-v2'); client.set_sparse_model('prithivida/Splade_PP_en_v1')"
