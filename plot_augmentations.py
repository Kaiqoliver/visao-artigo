import torch
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms

# 1. A sua classe de oclusão (copiada para cá para o script ser independente)
class RandomBackgroundInject(object):
    def __init__(self, p=1.0): # Forçando p=1.0 para garantir que apareça no plot
        self.p = p

    def __call__(self, tensor_img):
        if random.random() < self.p:
            mask = tensor_img.sum(dim=0) < 0.1 
            noise_background = torch.rand_like(tensor_img)
            tensor_img[:, mask] = noise_background[:, mask]
        return tensor_img

# 2. Defina o caminho de UMA imagem de exemplo (pegue uma folha qualquer do seu dataset)
# Troque este caminho por um arquivo real que exista na sua pasta
IMG_PATH = "/home/koliver/PlantVillage-Dataset/raw/color/Apple___Apple_scab/0b4a52e3-e15e-4117-b2e8-7cdb5dca3ce9___FREC_Scab 3137.JPG"

def main():
    # Carrega a imagem e aplica o Resize base
    img_original = Image.open(IMG_PATH).convert('RGB')
    resize = transforms.Resize((256, 256))
    img_base = resize(img_original)

    # 3. Define as transformações individuais
    # Nota: Probabilidades (p) alteradas para 1.0 para GARANTIR que o efeito apareça na imagem gerada
    
    # Geométricas
    t_rot = transforms.RandomRotation(degrees=(45, 45))(img_base) # Forçando 45 graus
    t_persp = transforms.RandomPerspective(distortion_scale=0.5, p=1.0)(img_base)
    t_flip = transforms.RandomHorizontalFlip(p=1.0)(img_base)

    # Cor e Ruído (PIL)
    t_color = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)(img_base)
    t_blur = transforms.GaussianBlur(kernel_size=(9, 9))(img_base)
    t_eq = ImageOps.equalize(img_base)

    # Tensores (Corte, Oclusão e Fundo)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Crop
    crop_tensor = transforms.RandomCrop(224)(img_base)
    t_crop = crop_tensor # Crop já retorna PIL se a entrada for PIL

    # Background Inject (Exige Tensor)
    tensor_base = to_tensor(img_base)
    tensor_bg = RandomBackgroundInject(p=1.0)(tensor_base.clone())
    t_bg = to_pil(tensor_bg)

    # Random Erasing (Exige Tensor)
    tensor_erase = transforms.RandomErasing(p=1.0, scale=(0.05, 0.1), value='random')(tensor_base.clone())
    t_erase = to_pil(tensor_erase)

    # 4. Monta o Dicionário para o Plot
    images = {
        "Original (256x256)": img_base,
        "Rotação (45°)": t_rot,
        "Perspectiva": t_persp,
        "Flip Horizontal": t_flip,
        "Jitter de Cor": t_color,
        "Desfoque Gaussiano": t_blur,
        "Equalização Hist.": t_eq,
        "Corte Aleatório (224)": t_crop,
        "Fundo Aleatório (Ruído)": t_bg,
        "Apagamento Aleatório": t_erase
    }

    # 5. Configura a figura do Matplotlib (alta resolução para artigo)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), dpi=300)
    axes = axes.flatten()

    for ax, (title, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off') # Remove os eixos x e y para ficar limpo

    plt.tight_layout()
    
    # Salva a imagem final
    output_filename = "grid_transformacoes_artigo.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Imagem salva com sucesso em: {output_filename}")

if __name__ == "__main__":
    main()