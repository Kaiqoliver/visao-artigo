import os
import torch
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.model import initialize_model

def evaluate_cross_dataset():
    # 1. Configurações de Caminhos e Dispositivo
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_path = '/home/koliver/visao-artigo/results/fine_tuning/model.pth'
    plantdoc_dir = '/home/koliver/PlantDoc-Dataset/train' # Pode mudar para test se o PlantDoc tiver
    output_csv = '/home/koliver/visao-artigo/results/fine_tuning/plantdoc_cross_eval.csv'

    print(f"Dispositivo selecionado: {device}")
    
    # 2. Mapeamento PlantVillage -> PlantDoc
    pv_to_plantdoc_map = {
        "Apple___Apple_scab": "Apple Scab Leaf",
        "Apple___Cedar_apple_rust": "Apple rust leaf",
        "Apple___healthy": "Apple leaf",
        "Blueberry___healthy": "Blueberry leaf",
        "Cherry_(including_sour)___healthy": "Cherry leaf",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray leaf spot",
        "Corn_(maize)___Common_rust_": "Corn rust leaf",
        "Corn_(maize)___Northern_Leaf_Blight": "Corn leaf blight",
        "Grape___Black_rot": "Grape leaf black rot",
        "Grape___healthy": "Grape leaf",
        "Peach___healthy": "Peach leaf",
        "Pepper,_bell___Bacterial_spot": "Bell_pepper leaf spot",
        "Pepper,_bell___healthy": "Bell_pepper leaf",
        "Potato___Early_blight": "Potato leaf early blight",
        "Potato___Late_blight": "Potato leaf late blight",
        "Potato___healthy": "Potato leaf",
        "Raspberry___healthy": "Raspberry leaf",
        "Soybean___healthy": "Soyabean leaf",
        "Squash___Powdery_mildew": "Squash Powdery mildew leaf",
        "Strawberry___healthy": "Strawberry leaf",
        "Tomato___Bacterial_spot": "Tomato leaf bacterial spot",
        "Tomato___Early_blight": "Tomato Early blight leaf",
        "Tomato___Late_blight": "Tomato leaf late blight",
        "Tomato___Leaf_Mold": "Tomato mold leaf",
        "Tomato___Septoria_leaf_spot": "Tomato Septoria leaf spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf yellow Virus",
        "Tomato___Tomato_mosaic_virus": "Tomato leaf mosaic virus",
        "Tomato___healthy": "Tomato leaf"
    }

    # 3. As 38 Classes do PlantVillage (Ordem exata que o modelo aprendeu)
    pv_classes = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", 
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", 
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", 
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", 
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", 
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
        "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]

    # 4. Inicializar e carregar os pesos do modelo (modo fine_tuning)
    # Passamos feature_extract=False e use_pretrained=True pois essa foi a arquitetura do fine_tuning
    model_ft, input_size = initialize_model(feature_extract=False, use_pretrained=True)
    model_ft.load_state_dict(torch.load(model_path, map_location=device))
    model_ft = model_ft.to(device)
    model_ft.eval() # Fundamental para desativar Dropout e BatchNorm em inferência

    # 5. Carregar o dataset do PlantDoc
    # Na avaliação, não fazemos Data Augmentation (sem RandomCrop ou Flip), apenas CenterCrop
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    plantdoc_dataset = datasets.ImageFolder(root=plantdoc_dir, transform=transform)
    dataloader = DataLoader(plantdoc_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    plantdoc_classes = plantdoc_dataset.classes
    print(f"Total de imagens no PlantDoc: {len(plantdoc_dataset)}")

    # 6. Loop de Avaliação
    y_true = []
    y_pred = []
    corrects = 0
    total = 0
    
    print("Iniciando inferência cross-dataset... (Isso pode levar alguns minutos)")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(preds)):
                # Classe real (Ground Truth) baseada nas pastas do PlantDoc
                gt_plantdoc_class = plantdoc_classes[labels[i].item()]
                
                # O que o modelo achou que era (Índice 0-37 -> Nome no PlantVillage)
                pred_pv_idx = preds[i].item()
                pred_pv_class = pv_classes[pred_pv_idx]
                
                # Tradução para o PlantDoc. Se o modelo prever algo como "Tomato Target Spot"
                # que não existe no PlantDoc, ele registra como "Unmapped_Prediction", o que
                # vai contar automaticamente (e corretamente) como um erro do modelo.
                pred_plantdoc_class = pv_to_plantdoc_map.get(pred_pv_class, "Unmapped_Prediction")
                
                y_true.append(gt_plantdoc_class)
                y_pred.append(pred_plantdoc_class)
                
                if gt_plantdoc_class == pred_plantdoc_class:
                    corrects += 1
                total += 1

    # 7. Resultados
    acc = corrects / total
    print("\n" + "="*50)
    print(f"🏆 Acurácia Real do Modelo no PlantDoc: {acc:.4f} ({(acc*100):.2f}%)")
    print("="*50)
    
    # Salvar DataFrame para plotar matriz de confusão e F1-score depois
    df = pd.DataFrame({'True_PlantDoc': y_true, 'Predicted_PlantDoc': y_pred})
    df.to_csv(output_csv, index=False)
    print(f"Resultados detalhados salvos em: {output_csv}")

if __name__ == "__main__":
    evaluate_cross_dataset()