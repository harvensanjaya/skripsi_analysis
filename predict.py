import tensorflow as tf
import numpy as np
import os
from collections import defaultdict
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load model yang sudah dilatih
MODEL_PATH = "heatmap_cnn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Folder gambar uji coba
TEST_IMAGE_DIR = "./test_images"
IMG_SIZE = (128, 128)

# Load label kelas dari dataset
DATASET_DIR = "./heatmap_dataset"
class_names = os.listdir(DATASET_DIR)  # Ambil nama kelas dari folder dataset

# Dictionary untuk menyimpan jumlah prediksi dan total confidence per kelas
class_counts = defaultdict(int)
class_confidences = defaultdict(float)

# Fungsi untuk melakukan prediksi pada gambar baru
def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)  # Load gambar
    img_array = img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Ubah ke batch (1, 128, 128, 3)

    predictions = model.predict(img_array)  # Prediksi kelas
    predicted_class = np.argmax(predictions)  # Indeks kelas dengan probabilitas tertinggi
    confidence = np.max(predictions)  # Nilai probabilitas tertinggi

    class_counts[class_names[predicted_class]] += 1
    class_confidences[class_names[predicted_class]] += confidence

    print(f"Gambar: {image_path}")
    print(f"Prediksi: {class_names[predicted_class]} (Confidence: {confidence:.2f})\n")

    # Tampilkan gambar
    plt.imshow(load_img(image_path))
    plt.axis("off")
    plt.title(f"Prediksi: {class_names[predicted_class]}\nConfidence: {confidence:.2f}")
    plt.show()

# Uji beberapa gambar di folder
def predict_all_images():
    for image_name in os.listdir(TEST_IMAGE_DIR):
        image_path = os.path.join(TEST_IMAGE_DIR, image_name)
        predict_image(image_path)
    
    # Hitung rata-rata confidence per kelas dan urutkan berdasarkan jumlah prediksi
    avg_confidences = {cls: class_confidences[cls] / class_counts[cls] for cls in class_counts}
    sorted_classes = sorted(class_counts.keys(), key=lambda cls: class_counts[cls], reverse=True)
    
    print("\n--- Statistik Prediksi ---")
    for cls in sorted_classes:
        print(f"Kelas: {cls} | Jumlah: {class_counts[cls]} | Confidence Rata-rata: {avg_confidences[cls]:.2f}")

# Jalankan prediksi pada semua gambar di folder test
if __name__ == "__main__":
    predict_all_images()
