import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import piq
from skimage import io

# Chemin d'accès au dossier contenant les images
img_folder = '/Users/medelbechir/Downloads/bdd'
compressed_folder = '/Users/medelbechir/Downloads/bdd_compressed'  # New folder to save compressed images

# Make sure the directory for the compressed images exists
os.makedirs(compressed_folder, exist_ok=True)

# Deux listes pour enregistrer les scores BRISQUE
original_brisque_scores = []
compressed_brisque_scores = []

# Fonction pour compresser l'image
def compress_image(img_path, N=60):
    #lecture d'une image
    image = plt.imread(img_path)

    #Décomposition en matrices R,G,B
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]

    #Décomposition en valeurs singulières et reconstruction pour chaque matrice
    for matrix in [R, G, B]:
        U, sigma, Vt = np.linalg.svd(matrix)
        matrix = np.matrix(U[:, :N]) * np.diag(sigma[:N]) * np.matrix(Vt[:N, :])

    #Reconstruction de l'image
    reconstimg=np.zeros(image.shape,dtype=int)
    reconstimg[:,:,0]=R
    reconstimg[:,:,1]=G
    reconstimg[:,:,2]=B

    return reconstimg

# Fonction pour calculer l'indice BRISQUE
@torch.no_grad()
def compute_brisque(img_path):
    tensor = torch.tensor(io.imread(img_path)).permute(2, 0, 1)
    x = tensor[None, ...] / 255.

    if torch.cuda.is_available():
        x = x.cuda()

    brisque_index: torch.Tensor = piq.brisque(x, data_range=1., reduction='none')

    return brisque_index.item()

# Boucle sur tous les fichiers dans le dossier
for img_file in os.listdir(img_folder):
    # Verifier si le fichier est une image
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(img_folder, img_file)

    # Compression
    compressed_img = compress_image(img_path)

    # Convert the image to uint8
    compressed_img = np.clip(compressed_img, 0, 255).astype('uint8')  # Clip values to range 0-255 and then convert

    # Save the compressed image to the new folder
    compressed_img_path = os.path.join(compressed_folder, img_file)
    plt.imsave(compressed_img_path, compressed_img)

    # Calculer l'indice BRISQUE pour l'image originale et l'image compressée
    original_brisque = compute_brisque(img_path)
    compressed_brisque = compute_brisque(compressed_img_path)

    # Enregistrer les scores BRISQUE dans les listes
    original_brisque_scores.append(original_brisque)
    compressed_brisque_scores.append(compressed_brisque)

    print(f"Original BRISQUE for {img_file}: {original_brisque:0.4f}")
    print(f"Compressed BRISQUE for {img_file}: {compressed_brisque:0.4f}")

# Dessiner les histogrammes des scores BRISQUE
plt.figure(figsize=(12, 6))
plt.hist(original_brisque_scores, bins=50, alpha=0.5, label='Original images')
plt.hist(compressed_brisque_scores, bins=50, alpha=0.5, label='Compressed images')
plt.title('BRISQUE score distributions')
plt.xlabel('BRISQUE score')
plt.ylabel('Number of images')
plt.legend()
plt.show()
