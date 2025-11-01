import torch
import numpy as np
import cv2 as cv
import os

def compute_mean_std(img_dir, use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.JPG', '.jpeg'))]

    sum_pixels = torch.zeros(3, device=device)
    sum_squared_pixels = torch.zeros(3, device=device)
    pixel_count = 0

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # OpenCV lê em BGR, converter para RGB
        image = image.astype(np.float32) / 255.0  # Normaliza corretamente para [0,1]

        # Converte para tensor e reorganiza os eixos para CHW
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).to(device)

        sum_pixels += image_tensor.sum(dim=(1, 2))  # Soma dos pixels por canal
        sum_squared_pixels += (image_tensor ** 2).sum(dim=(1, 2))  # Soma dos quadrados dos pixels por canal
        pixel_count += image_tensor.shape[1] * image_tensor.shape[2]  # Contagem total de pixels

    mean = sum_pixels / pixel_count
    variance = (sum_squared_pixels / pixel_count) - (mean ** 2)
    std = torch.sqrt(variance)

    return mean.cpu().numpy(), std.cpu().numpy()

# Substitua pelo caminho da sua pasta de imagens
img_dir = '/raid/dados/es111286/datasets/swiss_okutama/swiss/train/images'
mean, std = compute_mean_std(img_dir)
print(f"Mean: {mean}, Std: {std}")


