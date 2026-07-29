import os
import json
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import GroupShuffleSplit

def load_data(data_dir, input_size, batch_size=24):
    """
    Carrega o dataset PlantVillage local, mapeia as folhas usando leaf-map.json
    e cria splits de Treino (80%) e Val/Teste (20%) sem vazamento de dados.
    """
    # 1. Procura automaticamente a pasta das classes
    possible_paths = [
        data_dir,
        os.path.join(data_dir, "raw", "color"),
        os.path.join(data_dir, "color")
    ]
    
    target_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if len(subdirs) > 0:
                target_dir = path
                break

    if target_dir is None:
        raise FileNotFoundError(f"Não foi possível encontrar as pastas de classes em '{data_dir}'.")

    print(f"Carregando dataset local a partir de: {target_dir}")

    # 2. Carregar o mapeamento do JSON com o caminho exato
    json_path = "/home/koliver/PlantVillage-Dataset/leaf_grouping/leaf-map.json"
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Arquivo leaf-map.json não encontrado em {json_path}.")
        
    with open(json_path, 'r') as f:
        leaf_map = json.load(f)

    # 3. Transformações base
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=target_dir, transform=transform)

    # 4. Mapear cada imagem no dataset para seu grupo (ID da folha física)
    groups = []
    for path, _ in dataset.samples:
        filename = os.path.basename(path).split('.')[0].lower().strip()
        leaf_ids = leaf_map.get(filename, [filename])
        groups.append(leaf_ids[0])

    # 5. Dividir os grupos (80% Treino, 20% Teste/Validação)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(dataset.samples, groups=groups))

    print(f"Splits criados por grupo de folhas! Treino: {len(train_idx)} imagens | Teste (Val): {len(val_idx)} imagens")

    # 6. Criar os Subsets e DataLoaders
    image_datasets = {
        'train': Subset(dataset, train_idx),
        'val': Subset(dataset, val_idx)
    }

    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4) 
        for x in ['train', 'val']
    }

    return dataloaders_dict