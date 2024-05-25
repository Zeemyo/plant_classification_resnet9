from django.shortcuts import render
import torch
from torchvision import transforms
from PIL import Image
from .models import ResNet9
import matplotlib
matplotlib.use('Agg')  # Ganti backend ke Agg sebelum mengimpor pyplot
import matplotlib.pyplot as plt
import numpy as np
import os

# Inisialisasi model
num_classes = 14  # Sesuaikan dengan jumlah kelas yang sesuai
model = ResNet9(num_classes=num_classes)

# Memuat state dictionary ke model
state_dict = torch.load(r'C:\Main storage\Kuliah\Semester 8\Aplikasi Plant Leaf Detection\plant-disease-model-all.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Definisikan transformasi data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fungsi untuk melakukan inferensi pada gambar
def predict(image, model, transform):
    image_tensor = transform(image).unsqueeze(0)  # Tambahkan batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities

# Fungsi untuk membuat visualisasi
def create_visualization(image, predicted_class):
    # Pastikan direktori static ada
    visualization_dir = os.path.join('static')
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
        
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    visualization_path = os.path.join(visualization_dir, 'visualization.png')
    plt.savefig(visualization_path)
    plt.close()
    return visualization_path

# View untuk halaman utama
def index(request):
    if request.method == 'POST' and request.FILES['leaf_image']:
        leaf_image = request.FILES['leaf_image']
        image = Image.open(leaf_image).convert('RGB')
        predicted_class, probabilities = predict(image, model, transform)
        visualization_path = create_visualization(image, predicted_class)

        print(f"ini predicted {probabilities}")
        
        # Informasi akurasi dan kategori
        accuracy = 0.95  # Contoh akurasi, sesuaikan dengan akurasi model Anda
        categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 
                      'Category 5', 'Category 6', 'Category 7', 'Category 8', 
                      'Category 9', 'Category 10', 'Category 11', 'Category 12', 
                      'Category 13', 'Category 14']  # Sesuaikan dengan kategori daun Anda
        category = categories[predicted_class]
        
        return render(request, 'plant_classification_app/result.html', {
            'predicted_class': predicted_class,
            'probabilities': probabilities,
            'visualization_path': visualization_path,
            'accuracy': accuracy,
            'category': category
        })
    return render(request, 'plant_classification_app/index.html')
