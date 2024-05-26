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

# Inisialisasi model dengan jumlah kelas yang sesuai
num_classes = 10  # Sesuaikan dengan jumlah kelas yang sesuai
model = ResNet9(num_classes=num_classes)

# Memuat state dictionary dari file
state_dict = torch.load(r'C:\Main storage\Kuliah\Semester 8\Aplikasi Plant Leaf Detection\tomato-disease-model.pth', map_location=torch.device('cpu'))

# Menyesuaikan nama lapisan jika diperlukan
new_state_dict = {}
for k, v in state_dict.items():
    # Ganti nama lapisan jika ada perbedaan
    new_key = k.replace("res1.0.0", "res1.0").replace("res1.0.1", "res1.1").replace("res1.1.0", "res1.1").replace("res1.1.1", "res1.1").replace("res2.0.0", "res2.0").replace("res2.0.1", "res2.1").replace("res2.1.0", "res2.1").replace("res2.1.1", "res2.1")
    new_state_dict[new_key] = v

# Memuat state dictionary yang disesuaikan ke model
model.load_state_dict(new_state_dict)

# Menandakan bahwa model dalam mode evaluasi
model.eval()

# Definisikan transformasi data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
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
    # Konversi tensor probabilitas ke numpy array dan ke list agar mudah dibaca
    probabilities = probabilities.numpy().tolist()
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
        
        # Informasi kategori
        categories = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 
                      'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']  # Sesuaikan dengan kategori daun Anda
        category = categories[predicted_class]
        
        return render(request, 'plant_classification_app/result.html', {
            'predicted_class': predicted_class,
            'probabilities': probabilities,
            'visualization_path': visualization_path,
            'accuracy': 'N/A',  # Akurasi tidak tersedia karena kita tidak melakukan pelatihan
            'category': category,
            'categories': categories,  # Tambahkan ini untuk template
            'accuracy_graph': 'static/accuracy.png',  # Pastikan file ini ada
            'loss_graph': 'static/loss.png'  # Pastikan file ini ada
        })
    return render(request, 'plant_classification_app/index.html')
