import os
import sys
import torch
import pandas as pd
from PIL import ImageOps
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.model import initialize_model

def evaluate_model(model_path, output_csv, feature_extract=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    plantdoc_dir = '/home/koliver/PlantDoc-Dataset/train'

    print(f"\nAvaliando modelo em: {model_path}")
    print(f"Dispositivo selecionado: {device}")
    
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

    pv_classes = [
        "Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
        "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape_healthy", 
        "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", 
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", 
        "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", 
        "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", 
        "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", 
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]
    
    # Inicializa arquitetura de acordo com o modo (fine_tuning ou last_layer)
    model_ft, input_size = initialize_model(feature_extract=feature_extract, use_pretrained=True)
    
    if not os.path.exists(model_path):
        print(f"[AVISO] Arquivo de modelo não encontrado em: {model_path}. Pulando...")
        return False

    model_ft.load_state_dict(torch.load(model_path, map_location=device))
    model_ft = model_ft.to(device)
    model_ft.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda img: ImageOps.equalize(img)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    plantdoc_dataset = datasets.ImageFolder(root=plantdoc_dir, transform=transform)
    dataloader = DataLoader(plantdoc_dataset, batch_size=32, shuffle=False, num_workers=4)
    plantdoc_classes = plantdoc_dataset.classes

    y_true = []
    y_pred = []
    corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(preds)):
                gt_plantdoc_class = plantdoc_classes[labels[i].item()]
                pred_pv_idx = preds[i].item()
                pred_pv_class = pv_classes[pred_pv_idx]
                pred_plantdoc_class = pv_to_plantdoc_map.get(pred_pv_class, "Unmapped_Prediction")
                
                y_true.append(gt_plantdoc_class)
                y_pred.append(pred_plantdoc_class)
                
                if gt_plantdoc_class == pred_plantdoc_class:
                    corrects += 1
                total += 1

    acc = corrects / total if total > 0 else 0
    print(f"🏆 Acurácia Cross-Dataset (PlantDoc): {(acc*100):.2f}%")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame({'True_PlantDoc': y_true, 'Predicted_PlantDoc': y_pred})
    df.to_csv(output_csv, index=False)
    print(f"Salvo em: {output_csv}")
    return True

if __name__ == "__main__":
    # Permite rodar via terminal se quiser testar um avulso: python evaluate_cross.py <caminho_modelo> <caminho_csv> <feature_extract(True/False)>
    if len(sys.argv) >= 4:
        m_path = sys.argv[1]
        o_csv = sys.argv[2]
        f_ext = sys.argv[3].lower() == 'true'
        evaluate_model(m_path, o_csv, f_ext)