import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import open_clip
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import numpy as np

RESULTS_FOLDER = os.path.abspath('./results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

df = pd.read_pickle('image_embeddings.pickle')

model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')


#### Image-to-Image Search ####
def image_to_image(image_file):

    # This converts the image to a tensor
    image = preprocess(Image.open(image_file)).unsqueeze(0)

    # This calculates the query embedding
    query_embedding = F.normalize(model.encode_image(image))

    # Retrieve the image path that corresponds to the embedding in `df`
    # with the highest cosine similarity to query_embedding
    embeddings = torch.stack([torch.tensor(embedding) for embedding in df['embedding']])
    embeddings = F.normalize(embeddings, dim=-1)
    similarities = cosine_similarity(query_embedding.cpu().detach().numpy(), embeddings.cpu().detach().numpy())
    
    # Get the top 5 most similar images
    top_k_indices = similarities.argsort()[0][-5:][::-1]  # Top 5 indices sorted by similarity
    top_k_similarities = similarities[0][top_k_indices]

    results = []

    # Save and prepare the images for Flask
    for i, (idx, similarity) in enumerate(zip(top_k_indices, top_k_similarities), start=1):
        impath = df.iloc[idx]['file_name']
        impath = f"./coco_images_resized/{impath}"

        # Open and save the image to the results folder
        result_image_path = os.path.join(RESULTS_FOLDER, f'result_{i}.jpg')
        result_image = Image.open(impath)
        result_image.save(result_image_path)

        # Append the image URL and similarity score to results
        results.append({
            'image_url': f'/results/result_{i}.jpg',  # Flask URL to serve the image
            'similarity_score': similarity
        })

    return results



#### Text-to-Image Search ####
def text_to_image(text_query):
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer(text_query)
    query_embedding = F.normalize(model.encode_text(text))

    # Retrieve the image path that corresponds to the embedding in `df`
    # with the highest cosine similarity to query_embedding
    embeddings = torch.stack([torch.tensor(embedding) for embedding in df['embedding']])
    embeddings = F.normalize(embeddings, dim=-1)
    similarities = cosine_similarity(query_embedding.cpu().detach().numpy(), embeddings.cpu().detach().numpy())
    
    # Get the top 5 most similar images
    top_k_indices = similarities.argsort()[0][-5:][::-1]  # Top 5 indices sorted by similarity
    top_k_similarities = similarities[0][top_k_indices]

    results = []

    # Save and prepare the images for Flask
    for i, (idx, similarity) in enumerate(zip(top_k_indices, top_k_similarities), start=1):
        impath = df.iloc[idx]['file_name']
        impath = f"./coco_images_resized/{impath}"

        # Open and save the image to the results folder
        result_image_path = os.path.join(RESULTS_FOLDER, f'result_{i}.jpg')
        result_image = Image.open(impath)
        result_image.save(result_image_path)

        # Append the image URL and similarity score to results
        results.append({
            'image_url': f'/results/result_{i}.jpg',  # Flask URL to serve the image
            'similarity_score': similarity
        })

    return results



#### Hybrid Query ####
def hybrid_query(image_file, text, weight):
    image = preprocess(Image.open(image_file)).unsqueeze(0)
    image_query = F.normalize(model.encode_image(image))
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    texts = tokenizer(text)
    text_query = F.normalize(model.encode_text(texts))

    lam = weight # tune this

    query = F.normalize(lam * text_query + (1.0 - lam) * image_query)

    image_embeddings = torch.stack([torch.tensor(embedding) for embedding in df['embedding']])
    query_np = query.detach().cpu().numpy()
    image_embeddings_np = image_embeddings.detach().cpu().numpy()
    cosine_similarities = cosine_similarity(query_np, image_embeddings_np)
    
    # Get the top 5 most similar images
    top_k_indices = cosine_similarities.argsort()[0][-5:][::-1]  # Top 5 indices sorted by similarity
    top_k_similarities = cosine_similarities[0][top_k_indices]

    results = []

    # Save and prepare the images for Flask
    for i, (idx, similarity) in enumerate(zip(top_k_indices, top_k_similarities), start=1):
        impath = df.iloc[idx]['file_name']
        impath = f"./coco_images_resized/{impath}"

        # Open and save the image to the results folder
        result_image_path = os.path.join(RESULTS_FOLDER, f'result_{i}.jpg')
        result_image = Image.open(impath)
        result_image.save(result_image_path)

        # Append the image URL and similarity score to results
        results.append({
            'image_url': f'/results/result_{i}.jpg',  # Flask URL to serve the image
            'similarity_score': similarity
        })

    return results


#### PCA ####
def load_images(max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir('./coco_images_resized')):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join('./coco_images_resized', filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

def nearest_neighbors(query_embedding, embeddings, top_k=5):
    # query_embedding: The embedding of the query item (e.g., the query image) in the same dimensional space as the other embeddings.
    # embeddings: The dataset of embeddings that you want to search through for the nearest neighbors.
    # top_k: The number of most similar items (nearest neighbors) to return from the dataset.
    # Hint: flatten the "distances" array for convenience because its size would be (1,N)
    distances = euclidean_distances(query_embedding.reshape(1, -1), embeddings)
    distances = distances.flatten()
    nearest_indices = np.argsort(distances)[:top_k]
    return nearest_indices, distances[nearest_indices]

def pca_first_k(image_file, k):
    query_image = Image.open(image_file).convert('L').resize((224, 224))
    query_image_array = np.array(query_image, dtype=np.float32) / 255.0
    query_image_array = query_image_array.flatten()

    pca = PCA(n_components=k)
    train_images, train_image_names = load_images(max_images=2000, target_size=(224, 224))
    pca.fit(train_images)

    query_embedding = pca.transform([query_image_array])

    transform_images, transform_image_names = load_images(max_images=10000, target_size=(224, 224))
    reduced_embeddings = pca.transform(transform_images)

    # Calculate Euclidean distances to all other images' embeddings
    distances = euclidean_distances(query_embedding.reshape(1, -1), reduced_embeddings)
    distances = distances.flatten()

    # Get the indices of the top 5 nearest images
    top_k_indices, top_k_distances = nearest_neighbors(query_embedding, reduced_embeddings, top_k=k)

    # Prepare the results with image paths and distances
    results = []
    # Save and prepare the images for Flask
    for i, (idx, distance) in enumerate(zip(top_k_indices, top_k_distances), start=1):
        impath = df.iloc[idx]['file_name']
        impath = f"./coco_images_resized/{impath}"

        # Open and save the image to the results folder
        result_image_path = os.path.join(RESULTS_FOLDER, f'result_{i}.jpg')
        result_image = Image.open(impath)
        result_image.save(result_image_path)

        # Append the image URL and similarity score to results
        results.append({
            'image_url': f'/results/result_{i}.jpg',  # Flask URL to serve the image
            'distance': distance
        })

    return results

import shutil

def clear_results_folder():
    # Ensure the results folder exists
    if os.path.exists(RESULTS_FOLDER):
        # Loop through all files in the results folder and remove them
        for filename in os.listdir(RESULTS_FOLDER):
            file_path = os.path.join(RESULTS_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory if any
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
