import os
import json
import torch
import numpy as np
from PIL import ImageOps
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import GroupShuffleSplit

import torch
import random
from torchvision import transforms

class RandomBackgroundInject(object):
    """
    Substitui os pixels pretos (fundo da imagem segmentada) 
    por ruído aleatório ou uma cor aleatória.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor_img):
        if random.random() < self.p:
            # Encontra onde é fundo (considerando fundo preto = soma dos canais próxima de 0)
            # Como a imagem já é um tensor [C, H, W] variando de 0 a 1
            mask = tensor_img.sum(dim=0) < 0.1 
            
            # Cria um fundo de ruído aleatório
            noise_background = torch.rand_like(tensor_img)
            
            # Onde a máscara for verdadeira (fundo), colocamos o ruído
            tensor_img[:, mask] = noise_background[:, mask]
            
        return tensor_img

# ================= NOVA SEÇÃO: FILTRO DE CLASSES =================
VALID_CLASSES = sorted([
    "Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___healthy",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
])

class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # Pega as classes normais que o PyTorch achou na pasta
        classes, class_to_idx = super().find_classes(directory)
        # Filtra apenas as que estão na nossa lista válida
        classes = [c for c in classes if c in VALID_CLASSES]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx
# =================================================================

def load_data(data_dir, input_size, batch_size=24):
    possible_paths = [
        data_dir,
        os.path.join(data_dir, "raw", "color"),
        os.path.join(data_dir, "color")
    ]
    
    target_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            target_dir = path
            break

    if target_dir is None: raise FileNotFoundError(f"Não foi possível encontrar as pastas em '{data_dir}'.")

    json_path = "/home/koliver/PlantVillage-Dataset/leaf_grouping/leaf-map.json"
    with open(json_path, 'r') as f:
        leaf_map = json.load(f)

    # === PIPELINE COM EQUALIZAÇÃO DE HISTOGRAMA ===
    transform = transforms.Compose([
        # 1. Ajuste de tamanho base
        transforms.Resize((256, 256)),
        
        # 2. Transformações Geométricas (Ângulos e Perspectivas)
        transforms.RandomRotation(degrees=45),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Folhas também podem estar de cabeça para baixo
        
        # 3. Transformações de Cor e Ruído
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=(5, 9)),
            transforms.Lambda(lambda img: ImageOps.equalize(img)) # Sua equalização anterior como uma chance aleatória!
        ]),
        
        # 4. Cortes e Conversão
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        RandomBackgroundInject(p=0.7),
        
        # 5. Oclusão e Normalização (Operam direto no Tensor)
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = FilteredImageFolder(root=target_dir, transform=transform)
    print(f"Classes carregadas para treino: {len(dataset.classes)}") 

    groups = []
    for path, _ in dataset.samples:
        filename = os.path.basename(path).split('.')[0].lower().strip()
        leaf_ids = leaf_map.get(filename, [filename])
        groups.append(leaf_ids[0])

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(dataset.samples, groups=groups))

    image_datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4) 
        for x in ['train', 'val']
    }

    return dataloaders_dict