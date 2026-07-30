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
        
        # 5. Oclusão e Normalização (Operam direto no Tensor)
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])