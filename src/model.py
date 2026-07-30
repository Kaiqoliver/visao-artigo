from torchvision import models
from torchvision.models.googlenet import GoogLeNet_Weights
import torch.nn as nn
import torch.optim as optim

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(feature_extract, use_pretrained, num_classes=28):
    model_ft = None
    input_size = 224 # Tamanho esperado pelas camadas do GoogLeNet no PyTorch

    # Definindo o modelo GoogLeNet
    if use_pretrained:
        print("Using pretrained GoogLeNet!!")
        model_ft = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    else:
        model_ft = models.googlenet(weights=None)
    
    set_parameter_requires_grad(model_ft, feature_extract)
    
    # Tratando as saídas auxiliares do GoogLeNet (se existirem na inicialização)
    if hasattr(model_ft, 'aux1') and model_ft.aux1 is not None:
        model_ft.aux1.fc = nn.Linear(model_ft.aux1.fc.in_features, num_classes)
    if hasattr(model_ft, 'aux2') and model_ft.aux2 is not None:
        model_ft.aux2.fc = nn.Linear(model_ft.aux2.fc.in_features, num_classes)
        
    # Tratando a saída principal
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft, input_size


def define_optimizer(model_ft, device, feature_extract):
    model_ft = model_ft.to(device)

    params_to_update = []
    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # Hiperparâmetros baseados no artigo: SGD, lr=0.005, momentum=0.9, weight_decay=0.0005
    optimizer_ft = optim.SGD(params_to_update, lr=0.005, momentum=0.9, weight_decay=0.0005)
    return optimizer_ft