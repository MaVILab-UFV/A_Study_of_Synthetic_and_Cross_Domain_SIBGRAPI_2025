import numpy as np
from PIL import Image
import os

original_mapping = {
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
    (10, 10, 10): 10,
    (11, 11, 11): 11,
    (12, 12, 12): 12,
    (13, 13, 13): 13,
    (14, 14, 14): 14,
    (15, 15, 15): 15,
    (16, 16, 16): 16,
    (17, 17, 17): 17,
    (255, 255, 255): 18
}

new_mapping = {
    0: (255, 255, 255),    
    1: (135, 206, 250),    
    2: (0, 128, 0),       
    3: (107, 142, 35),     
    4: (255, 255, 0),      
    5: (255, 140, 0),     
    6: (255, 69, 0),       
    7: (255, 0, 0),        
    8: (178, 34, 34),      
    9: (139, 0, 0),        
    10: (165, 42, 42),     
    11: (255, 99, 71),     
    12: (205, 92, 92),     
    13: (240, 128, 128),  
    14: (250, 128, 114),  
    15: (255, 182, 193),   
    16: (255, 192, 203),  
    17: (245, 222, 179),  
    18: (51, 76, 76)       
}

def convert_image(input_image_path, output_image_path):
    img = Image.open(input_image_path).convert("RGB")
    data = np.array(img)
    
    converted_data = np.zeros_like(data)
    
    # Percorre o mapeamento original e faz a substituição pela nova cor
    for original_rgb, original_index in original_mapping.items():
        new_rgb = new_mapping[original_index]
        mask = np.all(data == original_rgb, axis=-1)
        converted_data[mask] = new_rgb

    # Cria a nova imagem
    converted_img = Image.fromarray(converted_data.astype(np.uint8))
    converted_img.save(output_image_path)
    print(f"Imagem convertida e salva em {output_image_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Percorre todos os arquivos da pasta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):  # Filtra apenas arquivos PNG
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            convert_image(input_image_path, output_image_path)

    print("Processamento concluído.")

input_folder = r'C:\Users\Juliana\Documents\mestrado\Pesquisa\SIBGRAPI_2025\img\hrnet_FOLD3_HRNet_V04_PesosPotsdam_CE_adam1e4_b36_img128_jaccard\gt'

output_folder = r'C:\Users\Juliana\Documents\mestrado\Pesquisa\SIBGRAPI_2025\img\hrnet_FOLD3_HRNet_V04_PesosPotsdam_CE_adam1e4_b36_img128_jaccard'
process_folder(input_folder, output_folder)
