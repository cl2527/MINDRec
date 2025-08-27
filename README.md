# Downloading microsoft MIND news dataset
export KAGGLE_USERNAME=YOUR_KAGGLE_USERNAME
export KAGGLE_KEY=YOUR_API_KEY

kaggle datasets download -d arashnic/mind-news-dataset -p raw_data/mind --unzip

#
TALLRec: Only single GPU! Efficient tuning framework to align Large Language Models with recommendation data

https://medium.com/@ichigo.v.gen12/tallrec-only-single-gpu-efficient-tuning-framework-to-align-llm-with-recommendation-data-5a8a8dba4874

# For deepseek R1 distilled:
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tok.pad_token = tok.eos_token  # Llama uses EOS as pad

bnb = BitsAndBytesConfig(load_in_8bit=True)  # int8 (QLoRA uses 4-bit; see below)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",            
    torch_dtype="auto",           
)

# fixing version incompability
pip uninstall -y bitsandbytes torch torchvision torchaudio

pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1"

pip install --no-cache-dir "bitsandbytes==0.43.3"
