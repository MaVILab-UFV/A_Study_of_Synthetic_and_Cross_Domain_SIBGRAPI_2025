from collections import defaultdict
import numpy as np
import os
import cv2

CUSTOM_COLORMAP = {
    (0, 0, 0): 0,
    (1, 1, 1): 1,   
    (2, 2, 2): 2,   
    (3, 3, 3): 3,    
    (4, 4, 4): 4,
    (5, 5, 5): 5,
    (6, 6, 6): 6,
    (7, 7, 7): 7,
    (8, 8, 8): 8,   
    (9, 9, 9): 9,   
}

def calculate_class_distribution(mask_dir, colormap):
    class_pixel_counts = defaultdict(int)

    mask_files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Converta a máscara para índices de classe
        for rgb, class_idx in colormap.items():
            mask_class = (mask[:, :, 0] == rgb[0]) & (mask[:, :, 1] == rgb[1]) & (mask[:, :, 2] == rgb[2])
            class_pixel_counts[class_idx] += np.sum(mask_class)

    total_pixels = sum(class_pixel_counts.values())
    class_frequencies = {cls: count / total_pixels for cls, count in class_pixel_counts.items()}

    return class_pixel_counts, class_frequencies

# Caminho das máscaras e colormap
mask_dir = '/raid/dados/es111286/datasets/swiss_okutama/swiss/train/ground_truth'

class_pixel_counts, class_frequencies = calculate_class_distribution(mask_dir, CUSTOM_COLORMAP)

print("Número de pixels por classe:")
for cls, count in class_pixel_counts.items():
    print(f"Classe {cls}: {count} pixels")

print("\nFrequência relativa por classe:")
for cls, freq in class_frequencies.items():
    print(f"Classe {cls}: {freq:.4f}")

