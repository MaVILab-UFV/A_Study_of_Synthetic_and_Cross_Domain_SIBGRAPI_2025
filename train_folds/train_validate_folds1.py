import os
import torch
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms
import cv2 as cv
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm 
import torch.nn as nn
from base_train.hrnet import HighResolutionNet 
import torchvision.transforms.functional as F
from PIL import Image
from custom_fold import CustomDataset 
import segmentation_models_pytorch as smp
import math
from base_train.dice_loss import DiceLoss
from sklearn.metrics import jaccard_score
from base_train.segmentationMetrics import SegmentationMetrics

class Dice_CrossEntropy_Loss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5):
        super(Dice_CrossEntropy_Loss, self).__init__()
        #self.dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.dice = DiceLoss(class_weights=weight)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

def load_best_model(model, best_model_path):
    if os.path.exists(best_model_path):
        print(f"Loading best saved model: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Model {best_model_path} not found!")
    return model

def check_weight_compatibility(model, state_dict):
    model_dict = model.state_dict()
    matched_keys = []
    unmatched_keys = []
    
    for k in state_dict.keys():
        if k in model_dict and model_dict[k].shape == state_dict[k].shape:
            matched_keys.append(k)
        else:
            unmatched_keys.append(k)
    
    print(f"Total number of weights in the pre-trained model: {len(state_dict.keys())}")
    print(f"Compatible weights loaded: {len(matched_keys)}")
    print(f"Incompatible weights: {len(unmatched_keys)}\n")

    if unmatched_keys:
        print("\n❌ Incompatible weights (not loaded):")
        print(unmatched_keys)

def train(model, optimizer, criterion, epochs, train_loader, val_loader, num_classes, model_path):
    print("Starting training...")   
    best_val_jaccard = 0.0
    best_model_jaccard = None    

    for epoch in range(epochs):
        model.train()        
        running_loss = 0.0
        running_accuracy = 0.0
        running_jaccard = 0.0
        jaccard_classes_epoch_sum = np.zeros(num_classes)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)  
            labels = labels.to(device)  
                        
            outputs = model(images)       
            
            logits = outputs[0] if isinstance(outputs, tuple) else outputs            
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                                    
            running_loss += loss.item()
                        
            accuracy = metrics.calculate_accuracy(outputs, labels)
            jaccard_metric, jaccard_classes = metrics.calculate_jaccard_sklearn(outputs, labels)   
            jaccard_classes_epoch_sum += np.array(jaccard_classes)
            running_accuracy += accuracy
            running_jaccard += jaccard_metric
                        
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / (batch_idx+1):.4f}, Accuracy: {running_accuracy / (batch_idx+1):.4f}, Jaccard: {running_jaccard / (batch_idx+1):.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        epoch_jaccard = running_jaccard / len(train_loader)
        jaccard_classes_epoch_avg = jaccard_classes_epoch_sum / len(train_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate_per_epoch', current_lr, epoch)
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
        writer.add_scalar('Jaccard/train', epoch_jaccard, epoch)
        
        for cls in range(num_classes):
            writer.add_scalar(f'Jaccard/train/class_{cls}', jaccard_classes_epoch_avg[cls], epoch)          
        
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_jaccard = 0.0
        val_jaccard_classes_epoch_sum = np.zeros(num_classes)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                loss = criterion(logits, labels)

                val_loss += loss.item()
                accuracy = metrics.calculate_accuracy(outputs, labels)
                jaccard_metric, jaccard_classes = metrics.calculate_jaccard_sklearn(outputs, labels)
                val_jaccard_classes_epoch_sum += np.array(jaccard_classes)
                val_accuracy += accuracy
                val_jaccard += jaccard_metric

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_jaccard /= len(val_loader)
        val_jaccard_classes_epoch_avg = val_jaccard_classes_epoch_sum / len(val_loader)

        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Jaccard/Validation', val_jaccard, epoch) 

        for cls in range(num_classes):
            writer.add_scalar(f'Jaccard/Validation/class_{cls}', val_jaccard_classes_epoch_avg[cls], epoch)               
        
        log_message = (
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_accuracy:.4f}, Train Jaccard: {epoch_jaccard:.4f},"
            f"Validation Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Val Jaccard {val_jaccard:.4f}\n"
        )
                
        print(log_message.strip())
        
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message)
        
        if val_jaccard > best_val_jaccard:
            best_val_jaccard = val_jaccard
            best_model_jaccard = f"best_model_jaccard_{TEST_VERSION}_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_jaccard)
            print(f"Best Jaccard model saved as '{best_model_jaccard}'.")        
             
        torch.save(model.state_dict(), model_path)
        print(f"Last model saved as " + model_path)

        scheduler.step(val_loss)
        #scheduler.step()

    print("Training completed.")
    return best_model_jaccard

def validate(model, criterion, val_loader, arquivo):
    # Validação ajustada
    val_loss = 0.0
    val_accuracy = np.zeros(n_classes)
    val_f1_score = np.zeros(n_classes)
    val_jaccard = np.zeros(n_classes)
    accuracy_counts = np.zeros(n_classes)
    jaccard_counts = np.zeros(n_classes)   
    val_precision = np.zeros(n_classes)  
    val_recall = np.zeros(n_classes)      
    model.eval()      

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validando"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)
            val_loss += loss.item()

            class_acc, acc_counts = metrics.calculate_class_accuracy(outputs, labels)
            mean_jaccard, class_jaccard = metrics.calculate_jaccard_sklearn(outputs, labels)
            
            val_accuracy += class_acc
            accuracy_counts += acc_counts
            val_jaccard += class_jaccard
            jaccard_counts += 1            

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            class_precision, class_recall = metrics.calculate_precision_recall(preds, labels_np)
            val_precision += class_precision
            val_recall += class_recall

            class_f1_scores = metrics.calculate_f1_score(preds, labels_np)
            val_f1_score += class_f1_scores            

    val_loss /= len(val_loader)
    val_accuracy /= np.maximum(accuracy_counts, 1)
    val_f1_score /= np.maximum(accuracy_counts, 1)
    val_jaccard /= np.maximum(jaccard_counts, 1)    

    val_precision /= np.maximum(accuracy_counts, 1)
    val_recall /= np.maximum(accuracy_counts, 1)

    # Salvar métricas
    metrics_path = arquivo
    with open(metrics_path, "w") as f:
        f.write(f"Val Loss: {val_loss:.4f}\n")
        for cls in range(n_classes):
            f.write(f"Class {cls} - Accuracy: {val_accuracy[cls]:.4f}, "
                f"F1 Score: {val_f1_score[cls]:.4f}, Jaccard: {val_jaccard[cls]:.4f}, "
                f"Precision: {val_precision[cls]:.4f}, Recall: {val_recall[cls]:.4f}\n")

    print(f"Metrics per class saved to {metrics_path}")

def convert_to_new_colormap(mask, colormap):
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, rgb in colormap.items():
        rgb_image[mask == class_idx] = rgb
    return rgb_image

def predict(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.44700035, 0.4861078, 0.39292008), std=(0.21602656, 0.19250856, 0.24037625)) #fold1
    ])
    
    print("Starting prediction and conversion on the validation images...")
    image_files = sorted([f for f in os.listdir(img_dir_test) if os.path.isfile(os.path.join(img_dir_test, f))])

    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processando imagens"):
            image_path = os.path.join(img_dir_test, image_file)

            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (128, 128), interpolation=cv.INTER_NEAREST) 
            input_tensor = transform(image).unsqueeze(0).to(device)

            output = model(input_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            _, pred_mask = torch.max(logits, dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            rgb_mask = convert_to_new_colormap(pred_mask, NEW_COLORMAP)

            output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_prediction_rgb.png")
            cv.imwrite(output_path, cv.cvtColor(rgb_mask, cv.COLOR_RGB2BGR))

    print(f"Predictions and conversions saved in {output_dir}")

best_model_accuracy = None
best_model_jaccard = None
TEST_VERSION = "FOLD1_Hrnet_V06_ImagesRGB_ImagesDepth_juntas_compesoLARSTreinadoDepth"

gpu_ids = [3]
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

log_dir = f"./logs_tensorboard/logs_{TEST_VERSION}"
writer = SummaryWriter(log_dir=f"./logs_tensorboard/logs_{TEST_VERSION}")

#fold1 depht e rgb
img_dir = "/raid/dados/es111286/datasets/swiss_okutama/regions/Fold1/train/images_rgbdepth" 
mask_dir = '/raid/dados/es111286/datasets/swiss_okutama/regions/Fold1/train/mask_rgbdepth'

img_dir_val = "/raid/dados/es111286/datasets/swiss_okutama/regions/Fold1/val/images_rgbdepth" 
mask_dir_val = '/raid/dados/es111286/datasets/swiss_okutama/regions/Fold1/val/mask_rgbdepth'

img_dir_test = "/raid/dados/es111286/datasets/swiss_okutama/regions/Fold1/test/images_rgbdepth" 
mask_dir_test = '/raid/dados/es111286/datasets/swiss_okutama/regions/Fold1/test/mask_rgbdepth'

# Colormap personalizado lars_14_mapas_reduzido17e9
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

NEW_COLORMAP = {
    0: (0, 0, 0),
    1: (1, 1, 1),   
    2: (2, 2, 2),   
    3: (3, 3, 3),    
    4: (4, 4, 4),
    5: (5, 5, 5),
    6: (6, 6, 6),
    7: (7, 7, 7),
    8: (8, 8, 8),   
    9: (9, 9, 9),   
}

metrics = SegmentationMetrics(num_classes=len(CUSTOM_COLORMAP))

print("Starting training dataset...")
train_dataset = CustomDataset(img_dir, mask_dir, CUSTOM_COLORMAP, debug_mode=False, crop_size=(512, 512), final_size=(128, 128), 
                              mean=(0.44700035, 0.4861078, 0.39292008), std=(0.21602656, 0.19250856, 0.24037625), train=True) #fold1                              
train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True)

print("Starting validation dataset...")
val_dataset = CustomDataset(img_dir_test, mask_dir_test, CUSTOM_COLORMAP, crop_size=(512, 512), final_size=(128, 128), 
                            mean=(0.44700035, 0.4861078, 0.39292008), std=(0.21602656, 0.19250856, 0.24037625)) #fold1                            
val_loader = DataLoader(val_dataset, batch_size=36, shuffle=False)

print("Iniciando modelo HRNet...")

in_channels = 3  
n_classes = len(CUSTOM_COLORMAP) 
timestamps = 1  
log_file_path = f"log_hrnet_{TEST_VERSION}.txt"

# Certifica-se de que o arquivo é criado ou limpo antes do treinamento
with open(log_file_path, "w") as log_file:
    log_file.write("Treinamento iniciado\n")

model = HighResolutionNet(in_channels=in_channels, n_classes=n_classes, timestamps=timestamps)

pretrained_weights_path = "weights/best_model_jaccard_v01_HRNet_LARS_DephtAnything_CEDICE_49.pth"


incompatibles = ['cls_head.weight', 'cls_head.bias', 'aux_head.3.weight', 'aux_head.3.bias']

if os.path.exists(pretrained_weights_path):
    print(f"Carregando pesos pré-treinados de {pretrained_weights_path}...")
    pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
    
    pretrained_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()
    check_weight_compatibility(model, pretrained_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    
    model.load_state_dict(model_dict, strict=False)    

    for param in model.parameters():
        param.requires_grad = True    

    print("✅ Pre-trained weights loaded successfully.")
else:
    print("No pre-trained weights found. Training from scratch.")

if torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) 

#fold1
class_frequencies = {
 0: 0.0550,
 1: 0.0158,
 2: 0.1403,
 3: 0.1652,
 4: 0.3791,
 5: 0.0140,
 6: 0.2113,
 7: 0.0123,
 8: 0.0070,
 9: 0.0000
}

ignored_classes = [9] 

class_weights = {
    cls: 0 if cls in ignored_classes else 1 / math.log(freq + 1.1)
    for cls, freq in class_frequencies.items()
}

max_weight = max(weight for cls, weight in class_weights.items() if cls not in ignored_classes)
normalized_weights = {
    cls: (weight / max_weight if cls not in ignored_classes else 0)
    for cls, weight in class_weights.items()
}

weights = torch.tensor(list(normalized_weights.values()), dtype=torch.float).to(device)

print("Normalized weights:", weights)
criterion = nn.CrossEntropyLoss(weight=weights)

model_path = f"last_model_hrnet_{TEST_VERSION}.pth"
best_model_jaccard = train(
  model, optimizer, criterion, epochs=50, train_loader=train_loader, 
   val_loader=val_loader, num_classes=len(CUSTOM_COLORMAP), model_path=model_path
)

if best_model_jaccard and os.path.exists(best_model_jaccard):
    model = load_best_model(model, best_model_jaccard)
    validate(model, criterion, val_loader, f"validation_hrnet_{TEST_VERSION}.txt")  
    predict(model, f"/raid/dados/es111286/repositorios/Projeto_Fazendas_Preparacao_Dataset_Japao/hrnet_{TEST_VERSION}_jaccard") 
else:
    print("No model was saved based on the best Jaccard score.")

