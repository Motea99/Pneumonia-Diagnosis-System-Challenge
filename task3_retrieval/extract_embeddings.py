import torch
import numpy as np
import faiss
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Step 1: Initialize Device and Model ---
# This ensures the code uses the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Step 2: Feature Extraction (Optimized for GPU) ---
features_list = []
print(f"⏳ Extracting visual features using: {device}...")

with torch.no_grad():
    for images, _ in tqdm(test_loader):
        # Move images to GPU to speed up the process
        images = images.to(device)
        
        # Ensure image has 3 channels for ViT compatibility
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Get visual embeddings from ViT
        outputs = model.vit(images, return_dict=True)
        
        # Extract features (Pooler output or CLS token)
        if outputs.pooler_output is not None:
            feat = outputs.pooler_output.cpu().numpy()
        else:
            feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        features_list.append(feat)

# Convert list to a single NumPy array for indexing
features = np.vstack(features_list).astype('float32')

# --- Step 3: Build Search Index using FAISS ---
# Using Euclidean distance (L2) for similarity search
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)

print(f"✅ Success! Indexed {index.ntotal} images.")

# --- Step 4: Visual Retrieval Function ---
def show_similar_cases(k=5):
    # Select a random patient from the test dataset
    idx = random.randint(0, len(test_dataset) - 1)
    img_tensor, label = test_dataset[idx]
    
    # Prepare query image and move to GPU
    query_input = img_tensor.unsqueeze(0).to(device)
    if query_input.shape[1] == 1:
        query_input = query_input.repeat(1, 3, 1, 1)
        
    # Extract feature vector for the query image
    with torch.no_grad():
        q_outputs = model.vit(query_input, return_dict=True)
        if q_outputs.pooler_output is not None:
            query_feat = q_outputs.pooler_output.cpu().numpy().astype('float32')
        else:
            query_feat = q_outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')

    # Search for the top-K similar cases in the database
    distances, indices = index.search(query_feat, k)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Display Target Image
    plt.subplot(1, k + 1, 1)
    plt.imshow(img_tensor[0], cmap='gray')
    plt.title(f"Target Patient\nLabel: {label[0]}", color='red', fontweight='bold')
    plt.axis('off')
    
    # Display Retrieved Similar Cases
    for i, m_idx in enumerate(indices[0]):
        m_img, m_label = test_dataset[m_idx]
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(m_img[0], cmap='gray')
        plt.title(f"Match {i+1}\nLabel: {m_label[0]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the retrieval engine
show_similar_cases(k=5)
