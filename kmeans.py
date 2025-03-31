import cv2
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats

# Direktori dataset heatmap
DATASET_DIR = "./heatmap_dataset"
IMG_SIZE = (128, 128)
K_CLUSTERS = 3  # Jumlah cluster yang ingin kita buat

def load_images(dataset_dir):
    images = []
    filenames = []
    for img_name in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        filenames.append(img_name)
    return images, filenames

def extract_red_regions(image, filename):
    # Konversi ke format RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ekstraksi kanal warna merah
    red_channel = image_rgb[:, :, 0]
    
    # Thresholding untuk mendeteksi area dengan intensitas merah tinggi
    _, binary_mask = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
    
    # Temukan area terhubung (connected components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    features = []
    for i in range(1, num_labels):  # Mulai dari 1 karena 0 adalah background
        x, y, w, h, area = stats[i]
        centroid_x, centroid_y = centroids[i]
        
        # Hitung intensitas merah rata-rata di area
        mean_red_intensity = np.max(red_channel[labels == i])
        
        # Rasio aspek area fokus (lebar / tinggi)
        aspect_ratio = w / h if h != 0 else 0
        
        features.append([centroid_x, centroid_y, area, mean_red_intensity, aspect_ratio])
    
    # Visualisasi proses ekstraksi
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_rgb)
    axs[0].set_title("Gambar Asli")
    axs[0].axis("off")
    
    axs[1].imshow(binary_mask, cmap='gray')
    axs[1].set_title("Deteksi Area Merah")
    axs[1].axis("off")
    
    axs[2].imshow(image_rgb)
    axs[2].scatter(centroids[1:, 0], centroids[1:, 1], c='black', marker='x')
    axs[2].set_title("Centroid Area Fokus Merah")
    axs[2].axis("off")
    
    plt.suptitle(f"Proses Deteksi - {filename}")
    plt.show()
    
    return features

def cluster_heatmap(features):
    # Konversi fitur menjadi array numpy
    features_array = np.array(features)
    
    # Gunakan hanya centroid untuk clustering (kolom 0 dan 1)
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_array[:, :2])

    # Simpan hasil clustering ke file CSV dan JSON
    df = pd.DataFrame(features_array, columns=['centroid_x', 'centroid_y', 'area', 'mean_red_intensity', 'aspect_ratio'])
    df['cluster'] = cluster_labels
    df.to_csv('clusters.csv', index=False)

    clusters_json = {
        "clusters": [
            {"x": float(row[0]), "y": float(row[1]), "area": float(row[2]), "intensity": float(row[3]), "cluster": int(row[5])}
            for row in df.itertuples(index=False)
        ]
    }
    with open('clusters.json', 'w') as f:
        json.dump(clusters_json, f, indent=4)

    print("Hasil clustering telah disimpan sebagai clusters.csv dan clusters.json")

    return cluster_labels, features_array

# Load dataset
images, filenames = load_images(DATASET_DIR)
all_features = []
all_filenames = []

# Ekstraksi fitur dari setiap gambar
for img, filename in zip(images, filenames):
    features = extract_red_regions(img, filename)
    if features:
        all_features.extend(features)
        all_filenames.extend([filename] * len(features))

# Clustering berdasarkan fitur lokasi fokus merah
cluster_labels, clustered_features = cluster_heatmap(all_features)

# Visualisasi hasil clustering
plt.figure(figsize=(8, 6))
plt.scatter(clustered_features[:, 0], clustered_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Centroid X')
plt.ylabel('Centroid Y')
plt.title('Clustering Area Fokus Merah dalam Heatmap')
plt.gca().invert_yaxis()  # Sesuaikan agar koordinat sesuai dengan gambar
plt.show()

# Statistik hasil clustering
cluster_counts = np.bincount(cluster_labels)
print("Jumlah elemen dalam setiap cluster:")
for i in range(len(cluster_counts)):
    print(f"Cluster {i}: {cluster_counts[i]} area fokus merah")
