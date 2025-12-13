# hindi song recommendation system
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# 1️⃣ Hindi song dataset
def load_hindi_song_dataset():
    return [
        "Tum Hi Ho: ROMANTIC EMOTIONAL LOVE",
        "Channa Mereya: SAD EMOTIONAL HEARTBREAK",
        "Kesariya: ROMANTIC SOULFUL MELODY",
        "Apna Bana Le: ROMANTIC SOFT LOVE",
        "Kal Ho Naa Ho: EMOTIONAL LIFE PHILOSOPHY",
        "Raabta: ROMANTIC DESTINY VIBE",
        "Tera Yaar Hoon Main: FRIENDSHIP EMOTIONAL",
        "Bekhayali: SAD INTENSE HEARTBREAK",
        "Shayad: ROMANTIC MELANCHOLY",
        "Ghungroo: PARTY DANCE ENERGY",
        "Badtameez Dil: PARTY FUN DANCE",
        "Ilahi: TRAVEL FEEL GOOD",
        "Zinda: MOTIVATIONAL ENERGY",
        "Agar Tum Saath Ho: SAD ROMANTIC EMOTIONAL"
    ]


# 2️⃣ Create embeddings
def embed_hindi_songs(hindi_songs_list):
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    embeddings = model.encode(
        hindi_songs_list,
        batch_size=12,
        max_length=1024
    )["dense_vecs"]

    return model, embeddings


# GPT-2 model for explanation
model_name = "gpt2"


# 3️⃣ Retrieve top matching songs
def retrieve_top_songs(query, model, hindi_songs_list, song_embeddings, top_k=4):
    query_emb = model.encode(
        query,
        batch_size=12,
        max_length=1024
    )["dense_vecs"]

    similarity_scores = {}

    for i, emb in enumerate(song_embeddings):
        sim = cosine_similarity([query_emb], [emb])[0][0]
        similarity_scores[hindi_songs_list[i]] = sim

    sorted_results = sorted(
        similarity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results[:top_k]


# 4️⃣ Generate explanation using GPT
def generate_response(query, retrieved_songs):
    generator = pipeline("text-generation", model=model_name)

    context = "\n".join([f"- {song}" for song, score in retrieved_songs])

    prompt = f"""
User Mood / Genre: {query}

Recommended Hindi Songs:
{context}

Explain why these songs match the user's mood:
"""

    response = generator(
        prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        truncation=True
    )

    return response[0]["generated_text"]


# 5️⃣ Main function
def main():
    hindi_songs_list = load_hindi_song_dataset()
    model, song_embeddings = embed_hindi_songs(hindi_songs_list)

    query = input("Enter mood or genre (e.g. romantic, sad, party): ")

    top_songs = retrieve_top_songs(
        query,
        model,
        hindi_songs_list,
        song_embeddings
    )

    print("\n🎯 Top recommended Hindi songs:\n")
    for song, score in top_songs:
        print(f"{song}  ---> Similarity: {score:.4f}")

    print("\n🧠 Generating explanation with GPT-2...\n")
    explanation = generate_response(query, top_songs)
    print(explanation)


if __name__ == "__main__":
    main()
