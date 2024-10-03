#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it
curl -fsSL https://ollama.com/install.sh | sh
python3 -c "import os; from sentence_transformers import SentenceTransformer; from transformers import AutoModelForMaskedLM, AutoTokenizer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); tokenizer = AutoTokenizer.from_pretrained('prithivida/Splade_PP_en_v1'); model = AutoModelForMaskedLM.from_pretrained('prithivida/Splade_PP_en_v1')"