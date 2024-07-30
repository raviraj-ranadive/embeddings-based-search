import os
import io
import ast
import nltk
import openai
import shutil
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize


# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "")

global SPELLINGS
global TAG_DICT


embedded_tags_df = pd.read_csv("./tagged_dataset/pil_embedded_tags.csv")

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words and word not in punctuation]
    return filtered_sentence


def get_top_n_similar_embeddings(new_embedding, tags_df, top_n=10):
    # Ensure the embeddings are in the correct format
    tag_embeddings = np.array(tags_df['tag_embeddings'].tolist())
    
    # Calculate the cosine similarity between the new embedding and the embeddings in the dataframe
    similarities = cosine_similarity([new_embedding], tag_embeddings)[0]
    
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    top_similarities = similarities[top_indices]
    top_embeddings = tags_df.iloc[top_indices]
    
    # Get the image file paths of the top_n most similar embeddings
    top_image_arrays = top_embeddings['numpy_arr_img'].tolist()
    
    return top_image_arrays, top_similarities



def embed_query(input_query):
    input_query = ', '.join(input_query)
    response = openai.embeddings.create(input=input_query, model="text-embedding-3-large")
    embeddings = response.data[0].embedding
    return embeddings


def get_sorted_similar_embeddings(new_embedding, tags_df):
    # Ensure the embeddings are in the correct format
    tag_embeddings = np.array(tags_df['tag_embeddings'].apply(ast.literal_eval).tolist())
    similarities = cosine_similarity([new_embedding], tag_embeddings)[0]
    
    # Get the image file paths
    image_paths = tags_df['image_path'].tolist()
    # image_arrays = tags_df['numpy_arr_img'].apply(ast.literal_eval).tolist()
    
    results = list(zip(image_paths, similarities))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Separate the sorted image paths and similarity scores
    sorted_image_paths = [result[0] for result in sorted_results][:30]
    sorted_similarities = [result[1] for result in sorted_results][:30]
  
    return sorted_image_paths, sorted_similarities


def process_text(input_text):

    # Create or clean the temporary directory
    temp_dir = "./temp_save"
    
    # Clean the contents of the temporary directory if it's not empty
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    downscaled_paths = []

    sentences = sent_tokenize(input_text)[:3]
    for sentence in sentences:
        filtered_sentence = remove_stopwords(sentence)
        input_query = ', '.join(filtered_sentence)
    input_embeddings = embed_query(input_query)
    sorted_images, sorted_similarities = get_sorted_similar_embeddings(input_embeddings, embedded_tags_df)

    for img_path in sorted_images:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        
        # Resize the image
        width, height = img.size
        if width >= height:
            new_width = 256
            new_height = int(256 * height / width)
        else:
            new_height = 256
            new_width = int(256 * width / height)
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save the resized image to temporary folder
        downscaled_path = os.path.join(temp_dir, img_name)
        resized_img.save(downscaled_path)
        downscaled_paths.append(downscaled_path)
    
    return downscaled_paths
    


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