# 🚀 LLaMA-2 Mini Pretraining on Custom Dataset

This repository contains a fully working implementation of **pretraining a miniature LLaMA-2 style model** on a custom text dataset using HuggingFace Transformers.

The project trains a downsized LLaMA architecture on your own data and then uses it for text generation.

---

## 📌 Features

- 🔹 Train a **custom small LLaMA model** from scratch  
- 🔹 Uses **AutoTokenizer** + LLaMA architecture  
- 🔹 Works on **CPU** (slow but possible) or GPU  
- 🔹 Supports any `.txt` dataset  
- 🔹 Saves trained model + tokenizer  
- 🔹 Includes text generation pipeline  

---

## 📂 Project Structure

generative_ai/
│── pretraining.py # Main script
│── first.txt # Custom dataset
│── llama2-mini-model/ # Saved model + tokenizer
│── README.md # Documentation

yaml
Copy code

---

## 🧠 Requirements

Install the required libraries:

```bash
pip install transformers datasets accelerate sentencepiece
If using LLaMA-2 (gated model), login first:

bash
Copy code
huggingface-cli login
📝 Dataset
Place your dataset at:

makefile
Copy code
C:\generative_ai\first.txt
Example content:

csharp
Copy code
Cricket is a popular sport played between two teams...
Your model will learn patterns from this file.

▶️ Running the Training
Run:

bash
Copy code
python pretraining.py
Expected output:

vbnet
Copy code
train_loss: 9.11
epoch: 3.0
Model saved to ./llama2-mini-model
💾 Output Files
The trained model is saved at:

pgsql
Copy code
llama2-mini-model/
│── config.json
│── pytorch_model.bin
│── tokenizer.json
│── tokenizer_config.json
│── special_tokens_map.json
You can load them anytime for inference.

🧪 Inference Example
python
Copy code
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./llama2-mini-model",
    tokenizer="./llama2-mini-model"
)

prompt = "What is cricket?"
output = generator(prompt, max_length=100)

print(output[0]["generated_text"])
⚠️ Windows Terminal Unicode Fix
If your terminal throws:

vbnet
Copy code
UnicodeEncodeError: 'charmap' codec can't encode character
Add:

python
Copy code
import sys
sys.stdout.reconfigure(encoding='utf-8')
⭐ Future Improvements
Add LoRA fine-tuning

Add GPU training support

Add multiple dataset support

Create a Chat UI around the model

📄 License
MIT License — free to use and modify.

