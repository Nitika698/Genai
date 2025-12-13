# 🎵 Hindi Song Recommendation System (AI-Powered)

## 📌 Project Overview
The **Hindi Song Recommendation System** is an AI-based application that recommends Hindi songs based on a user’s **mood or genre**.  
It uses **semantic embeddings** to understand user intent and **generative AI** to explain why the recommended songs match the given mood.

This project demonstrates a practical combination of **Information Retrieval** and **Natural Language Generation (NLG)** using modern AI models.

---

## 🚀 Features
- 🎧 Mood-based Hindi song recommendations  
- 🧠 Semantic similarity using **BGE-M3 embeddings**
- 📊 Cosine similarity for accurate ranking
- ✍️ Natural-language explanation using **GPT-2**
- ⚡ Lightweight and fast execution
- 🔧 Easy to extend with more songs or languages

---

## 🧠 Technologies Used
- **Python 3.10+**
- **FlagEmbedding (BGE-M3 Model)**
- **Scikit-learn**
- **Hugging Face Transformers**
- **PyTorch**

---

## 📂 Project Structure
.
├── task.py
├── README.md

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Create and activate virtual environment
```bash
python -m venv genai310
genai310\Scripts\activate   # Windows
2️⃣ Install required dependencies
bash
Copy code
python -m pip install --upgrade pip
python -m pip install FlagEmbedding transformers scikit-learn torch
▶️ How to Run
bash
Copy code
python task.py
📝 Example Input
java
Copy code
Enter mood or genre (e.g. romantic, sad, party): romantic
📊 Example Output
yaml
Copy code
🎯 Top recommended Hindi songs:

Tum Hi Ho: ROMANTIC EMOTIONAL LOVE  ---> Similarity: 0.83
Kesariya: ROMANTIC SOULFUL MELODY ---> Similarity: 0.79
Raabta: ROMANTIC DESTINY VIBE ---> Similarity: 0.76
Apna Bana Le: ROMANTIC SOFT LOVE ---> Similarity: 0.74
An AI-generated explanation is also provided describing why these songs match the user's mood.

🔮 Future Enhancements
🔍 Use FAISS for scalable vector search

🎶 Integrate Spotify API for real song metadata

🌐 Add multilingual support (Punjabi, English, etc.)

🖥 Build a Streamlit or Django web interface

🤖 Replace GPT-2 with LLaMA / Mistral

📜 License
This project is open-source and available for educational and learning purposes.
