import os
import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import torchvision.transforms.functional as F
import random

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, colormap, crop_size=(480, 480), final_size=(128, 128), 
                 debug_mode=False, debug_output_dir="debug_samples",                  
                 mean=(0.33281243, 0.37414336, 0.28102675), std=(0.16513239, 0.16471149, 0.14268468), train=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.colormap = colormap
        self.crop_size = crop_size
        self.final_size = final_size
        self.debug_mode = debug_mode
        self.debug_output_dir = debug_output_dir
        self.train = train
        self.mean = mean 
        self.std = std       
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))],
            key=lambda x: int(x.split('.')[0].split('_')[-1])
        )
        self.mask_files = sorted(
            [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))],
            key=lambda x: int(x.split('.')[0].split('_')[-1])
        )

        if len(self.img_files) != len(self.mask_files):
            print(f"Atenção: Número de imagens ({len(self.img_files)}) e máscaras ({len(self.mask_files)}) não coincidem!")
        self.img_files, self.mask_files = self._filter_pairs(self.img_files, self.mask_files)
        print(f"Total de imagens: {len(self.img_files)}, Total de máscaras: {len(self.mask_files)}")

        # Criar pasta de debug se necessário
        if self.debug_mode:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            self.debug_samples = random.sample(list(zip(self.img_files, self.mask_files)), min(20, len(self.img_files)))        

    def random_crop(self, image, mask, size):
        """Realiza um corte aleatório na imagem e na máscara."""
        height, width = image.shape[:2]
        crop_height, crop_width = size

        if height < crop_height or width < crop_width:
            raise ValueError("A imagem ou máscara é menor que o tamanho do crop.")

        top = np.random.randint(0, height - crop_height + 1)
        left = np.random.randint(0, width - crop_width + 1)

        cropped_image = image[top:top + crop_height, left:left + crop_width]
        cropped_mask = mask[top:top + crop_height, left:left + crop_width]

        return cropped_image, cropped_mask

    def random_flip(self, image, mask):
        flip_type = random.choice([None, 0, 1, -1])  # None: sem flip, 0: vertical, 1: horizontal, -1: ambos
        if flip_type is not None:
            image = cv.flip(image, flip_type)
            mask = cv.flip(mask, flip_type)
        return image, mask

    def random_rotate(self, image, mask):
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            image = cv.rotate(image, {90: cv.ROTATE_90_CLOCKWISE,
                                    180: cv.ROTATE_180,
                                    270: cv.ROTATE_90_COUNTERCLOCKWISE}[angle])
            mask = cv.rotate(mask, {90: cv.ROTATE_90_CLOCKWISE,
                                    180: cv.ROTATE_180,
                                    270: cv.ROTATE_90_COUNTERCLOCKWISE}[angle])
        return image, mask

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image_file = self.img_files[index]
        mask_file = self.mask_files[index]
        image_path = os.path.join(self.img_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Carregar a imagem e a máscara
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

        # Crop
        image, mask = self.random_crop(image, mask, self.crop_size)

        if self.train:
             # Data augmentation  
            image, mask = self.random_flip(image, mask)  
            #print("✅ Random flip.") 
            image, mask = self.random_rotate(image, mask)  
            #print("✅ Random rotate.")         

        # Salvar imagens após crop, antes do resize
        if self.debug_mode and (image_file, mask_file) in self.debug_samples:
            base_name = os.path.splitext(image_file)[0]
            image_cropped_path = os.path.join(self.debug_output_dir, f"{base_name}_image_cropped.png")
            mask_cropped_path = os.path.join(self.debug_output_dir, f"{base_name}_mask_cropped.png")
            cv.imwrite(image_cropped_path, image)
            cv.imwrite(mask_cropped_path, mask)  # máscara já está em RGB

        # Resize final
        image_resized = cv.resize(image, self.final_size, interpolation=cv.INTER_NEAREST)
        mask_resized = cv.resize(mask, self.final_size, interpolation=cv.INTER_NEAREST)

        # Salvar imagens após resize
        if self.debug_mode and (image_file, mask_file) in self.debug_samples:
            image_resized_path = os.path.join(self.debug_output_dir, f"{base_name}_image_resized.png")
            mask_resized_path = os.path.join(self.debug_output_dir, f"{base_name}_mask_resized.png")
            cv.imwrite(image_resized_path, image_resized)
            cv.imwrite(mask_resized_path, mask_resized)

        # Para o modelo
        image_tensor = F.to_tensor(image_resized)
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)
        mask_tensor = self.encode_segmap(np.array(mask_resized))

        return image_tensor, mask_tensor.clone().detach().long()

    def encode_segmap(self, mask):
        """Converte a máscara RGB para valores de classes."""
        mask_class = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)
        for rgb, class_idx in self.colormap.items():
            mask_class[(mask[:, :, 0] == rgb[0]) & (mask[:, :, 1] == rgb[1]) & (mask[:, :, 2] == rgb[2])] = class_idx
        return mask_class

    def _filter_pairs(self, img_files, mask_files):
        filtered_img_files = []
        filtered_mask_files = []
        for img, mask in zip(img_files, mask_files):
            img_path = os.path.join(self.img_dir, img)
            mask_path = os.path.join(self.mask_dir, mask)
            if os.path.exists(img_path) and os.path.exists(mask_path):
                filtered_img_files.append(img)
                filtered_mask_files.append(mask)
            else:
                print(f"Imagem ou segmentação não encontrada para: {img} ou {mask}")
        return filtered_img_files, filtered_mask_files
