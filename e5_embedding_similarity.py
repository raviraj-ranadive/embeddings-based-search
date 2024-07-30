import os
import ast
import nltk
import torch
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from string import punctuation
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('intfloat/multilingual-e5-small').to(device)


embedded_tags_df = pd.read_csv("./test-e5/e5_embedded.csv")

# Function to remove stopwords from a sentence
def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words and word not in punctuation]
    return filtered_sentence

# Function to embed a query
def embed_query(input_query):
    input_query = ', '.join(input_query)
    embeddings = model.encode(input_query)
    embeddings = list(embeddings)
    return embeddings

def get_icon_path(image_path):
    sub_path, image_name = os.path.split(image_path)
    image_uuid, ext = os.path.splitext(image_name)
    updated_image_name = image_uuid + "_icon" + ext
    final_image_path = os.path.join(sub_path, updated_image_name)
    return final_image_path

# Function to get sorted similar embeddings
def get_sorted_similar_embeddings(new_embedding, tags_df):
    tag_embeddings = np.array(tags_df['tag_embeddings'].apply(ast.literal_eval)).tolist()
    similarities = cosine_similarity([new_embedding], tag_embeddings)[0]
    image_paths = tags_df['image_path'].tolist()
    results = list(zip(image_paths, similarities))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_image_paths = [result[0] for result in sorted_results]
    sorted_similarities = [result[1] for result in sorted_results]
    for img, score in zip(sorted_image_paths, sorted_similarities):
        print(img,   ":", score)

    return sorted_image_paths, sorted_similarities

# Function to process text input and retrieve images
def process_text(input_text):
    sentences = sent_tokenize(input_text)[:3]
    input_query = ''
    for sentence in sentences:
        filtered_sentence = remove_stopwords(sentence)
        input_query += ', '.join(filtered_sentence) + ', '
    input_embeddings = embed_query(input_query)
    sorted_images, sorted_similarities = get_sorted_similar_embeddings(input_embeddings, embedded_tags_df)
    final_output = [get_icon_path(im_path) for im_path in sorted_images]
    return final_output

# Gradio interface setup
with gr.Blocks() as demo:
    input_textbox = gr.Textbox(label="Enter text for image search")
    gallery = gr.Gallery(
        label="Most similar images", show_label=False, elem_id="gallery",
        columns=3, rows=10, object_fit="contain", height="auto",
    )
    btn = gr.Button("Search images", scale=0)

    btn.click(process_text, inputs=input_textbox, outputs=gallery)

if __name__ == "__main__":
    demo.launch(share=True)
