from __future__ import print_function 
from __future__ import division
import torch
print("PyTorch Version: ", torch.__version__)
import torchvision
print("Torchvision Version: ", torchvision.__version__)
import argparse
from src.save import *
from src.data import load_data
from src.train import train_model
from src.model import *

# Função para converter as strings do terminal ("True"/"False") em Booleanos reais do Python
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Esperado um valor booleano (True/False).')


def main(epochs, feature_extract, use_pretrained, working_mode, data_dir, output_dir, aug_geom, aug_color, aug_occlus):
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("The selected device is:", device)
    
    # Initialize the model for this run
    model_ft, input_size = initialize_model(feature_extract, use_pretrained)
    print(model_ft)
    
    # Load the data (passando os novos parâmetros de Augmentation)
    dataloaders_dict = load_data(
        data_dir, 
        input_size, 
        batch_size=24, 
        aug_geom=aug_geom, 
        aug_color=aug_color, 
        aug_occlus=aug_occlus
    )
    
    # Define the optimizer
    optimizer_ft = define_optimizer(model_ft, device, feature_extract)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate
    results = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, working_mode, num_epochs=epochs)

    # Save outputs (Agora vai salvar na pasta específica que o script .sh mandar)
    save_model(results['model_ft'], output_dir, working_mode)
    save_val_acc_history(results['val_acc_history'], output_dir, working_mode)
    save_val_loss_history(results['val_loss_history'], output_dir, working_mode)
    save_train_acc_history(results['train_acc_history'], output_dir, working_mode)
    save_train_loss_history(results['train_loss_history'], output_dir, working_mode)
    save_confusion_matrix(results['best_true'], results['best_preds'], output_dir, working_mode)
    
    print(f"############################################## {working_mode} ###############################################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with different modes')

    # Add the arguments
    parser.add_argument('--epochs', type=int, default=15, help='The number of epochs')
    parser.add_argument('--feature_extract', type=str, default="False", help='Flag for feature extracting.')
    parser.add_argument('--use_pretrained', type=str, default="False", help='Use pretrained model or not')
    parser.add_argument('--dataset_dir', type=str, default='/home/koliver/PlantVillage-Dataset/raw/color')
    parser.add_argument('--output_dir', type=str, default='/home/koliver/visao-artigo/results')
    
    # Argumentos do Estudo de Ablação
    parser.add_argument('--aug_geom', type=str2bool, default=False, help='Ativa rotação, perspectiva, flip')
    parser.add_argument('--aug_color', type=str2bool, default=False, help='Ativa distorção de cor e brilho')
    parser.add_argument('--aug_occlus', type=str2bool, default=False, help='Ativa oclusão (RandomErasing)')

    args = parser.parse_args()
    
    feature_extract_bool = (args.feature_extract == "True")
    use_pretrained_bool = (args.use_pretrained == "True")

    print("The selected epochs is:", args.epochs)
    print("The selected feature_extract is:", feature_extract_bool)
    print("The selected use_pretrained is:", use_pretrained_bool)
    print("Augmentation -> Geometria:", args.aug_geom, "| Cor:", args.aug_color, "| Oclusão:", args.aug_occlus)
    
    working_mode = ""
    if feature_extract_bool and use_pretrained_bool:
        working_mode = "last_layer"
    elif not feature_extract_bool and use_pretrained_bool:
        working_mode = "fine_tuning"
    elif not feature_extract_bool and not use_pretrained_bool:
        working_mode = "from_scratch"

    print("The selected mode is:", working_mode)
    print("Output directory is:", args.output_dir)
    
    main(
        args.epochs, 
        feature_extract_bool, 
        use_pretrained_bool, 
        working_mode, 
        args.dataset_dir, 
        args.output_dir,
        args.aug_geom,
        args.aug_color,
        args.aug_occlus
    )