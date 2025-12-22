from transformers import AutoTokenizer

TOKENIZER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)

def get_tokenizer():
    return _tokenizer