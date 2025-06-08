import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import gradio as gr

# Load data
books = pd.read_csv("books_with_emotion.csv")

# Fix image URLs
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "image.png",
    books["large_thumbnail"]
)

# Document processing
raw_documents = TextLoader("tagged_description.txt",encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator="\n")
documents= text_splitter.split_documents(raw_documents)

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths and embedding function
persist_directory = r".\chroma_db"  # Raw string for Windows paths
embedding_function = HuggingFaceEmbeddings(  # Must match the original model used
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# Load existing Chroma DB
db_books = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)


# Recommendation function
def retrieve_semantic_recommendation(query: str, category: str = None, tone: str = None,
                                     initial_top_k: int = 50, final_top_k=16) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)

    books_list = []
    for doc, score in recs:  # Directly unpack in the loop
     books_list.append(int(doc.page_content.split()[0].lstrip('"').strip()))



    books_recs = books[books["isbn13"].isin(books_list)]

    if category and category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)

    if tone and tone != "All":
        tone_map = {
            "Happy": "joy",
            "Surprising": "surprising",
            "angry": "anger",
            "suspenseful": "fear",
            "sad": "sadness"
        }
        if tone in tone_map:
            books_recs = books_recs.sort_values(by=tone_map[tone], ascending=False)

    return books_recs.head(final_top_k)


# Display function for GUI
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendation(query, category, tone)

    if recommendations.empty:
        return [("image.png", "No matches found. Try a different query!")]

    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc = " ".join(description.split()[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 1:
            authors_str = authors[0]
        elif len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        else:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"

        caption = f"{row['title']} by {authors_str}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))

    return results


# GUI setup
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "angry", "suspenseful", "sad"]

with gr.Blocks(theme='earneleh/paris') as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Describe your ideal book:",
                                placeholder="e.g., A thrilling mystery novel")
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Mood", value="All")
        submit_btn = gr.Button("Find Recommendations")

    gr.Markdown("## 📖 Recommendations")
    output = gr.Gallery(label="", columns=4, rows=2)

    submit_btn.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()


recs = db_books.similarity_search_with_score(user_query, k=20)
print("Similarity search results:", recs)

