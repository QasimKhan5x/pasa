import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

os.environ['HF_HOME'] = '/project/models'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('prithivida/Splade_PP_en_v1', token=os.environ["HUGGINGFACE_TOKEN"])
model = AutoModelForMaskedLM.from_pretrained('prithivida/Splade_PP_en_v1', token=os.environ["HUGGINGFACE_TOKEN"])
