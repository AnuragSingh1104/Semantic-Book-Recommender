# 📚 Semantic Book Recommender

Semantic Book Recommender is an interactive AI-powered web application that helps users discover books tailored to their interests, genres, and emotional tone. Leveraging modern NLP embeddings and vector search (ChromaDB), it provides recommendations based on the *meaning* of your query, not just keywords.

---

## 🚀 Features

- **Semantic Search:** Find books by describing your ideal story in natural language.
- **Emotion Filtering:** Filter recommendations by mood (Happy, Surprising, Angry, Suspenseful, Sad).
- **Category Selection:** Choose from a wide range of genres.
- **Interactive UI:** Clean, modern Gradio interface with instant results and example queries.
- **Image Gallery:** See book covers and short descriptions for each recommendation.

---

## 🛠️ How It Works

1. **Data Loading:**  
   Loads book metadata (ISBN, title, authors, categories, emotion scores, image URLs) from a CSV file.

2. **Embedding & Vector Store:**  
   Book descriptions are embedded using a SentenceTransformer model (`BAAI/bge-large-en-v1.5`). Embeddings are stored and queried using ChromaDB[2].

3. **User Query:**  
   Users enter a description, select a category and mood, and submit their request.

4. **Semantic Retrieval:**  
   The app performs a similarity search in the vector database to find books whose descriptions best match the user's query[2].

5. **Filtering & Ranking:**  
   Results are filtered by category and sorted by the selected emotional tone.

6. **Display:**  
   The top recommendations are shown in a gallery with cover images and truncated descriptions[1].

---

## 📦 File Structure

├── books_with_emotion.csv # Book metadata with emotion scores
├── tagged_description.txt # Book descriptions, one per line
├── gradio-dash.py # Main app script
├── requirements.txt # Python dependencies
├── chroma_db/ # Chroma vector database (auto-generated)
└── README.md # This file



---

## ⚡ Installation

1. **Clone the repository:**
git clone https://github.com/AnuragSingh1104/Semantic-Book-Recommender.git
cd Semantic-Book-Recommender

2. **Create and activate a virtual environment:**
python -m venv .venv

Windows
..venv\Scripts\activate

macOS/Linux
source .venv/bin/activate

3. **Install dependencies:**
pip install -r requirements.txt

text

---

## ▶️ Usage

1. **Prepare Data:**
- Ensure `books_with_emotion.csv` and `tagged_description.txt` are in the project directory.

2. **Run the app:**
python gradio-dash.py

text

3. **Open your browser:**  
Go to [http://127.0.0.1:7860](http://127.0.0.1:7860) to use the app.

---

## 📝 Requirements

- Python 3.8+
- gradio
- pandas
- numpy
- langchain
- langchain-community
- langchain-huggingface
- langchain-chroma
- sentence-transformers

---

## ✨ Credits

- Built by [Anurag Singh](https://github.com/AnuragSingh1104)
- Book data: Open Library and other sources
- Powered by [Gradio](https://gradio.app/) and [ChromaDB](https://www.trychroma.com/)[2]

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

Thanks to the open-source community for the tools and datasets that made this project possible.

---

*Happy reading! 📚*
