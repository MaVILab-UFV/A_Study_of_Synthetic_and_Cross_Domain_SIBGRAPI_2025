#IMPORTS
import os
import torch
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms
import cv2 as cv
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm 
import torch.nn as nn
from hrnet import HighResolutionNet 
import torchvision.transforms.functional as F
from PIL import Image
from custom_dataset_hrnet import CustomDataset 
import segmentation_models_pytorch as smp
import math
from sklearn.metrics import jaccard_score
from segmentationMetrics import SegmentationMetrics
from datetime import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
import kornia.augmentation as K
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

#Dice&CrossEntropy Loss
class Dice_CrossEntropy_Loss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5):
        super(Dice_CrossEntropy_Loss, self).__init__()
        self.dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        #self.dice = DiceLoss(class_weights=weight)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

def save_image_tensor(tensor_img, path, mean, std):
    # tensor_img: 3 x H x W
    img = tensor_img.detach().cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  # desnormaliza
    np_img = img.numpy()
    np_img = (np.clip(np.transpose(np_img, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
    plt.imsave(path, np_img)

# Função auxiliar para converter uma máscara de classes (inteiros) para imagem RGB
def convert_mask_to_rgb(mask, colormap):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for rgb, label in colormap.items():
        rgb_mask[mask == label] = rgb
    return rgb_mask


def decode_mask_tensor(mask_tensor, colormap):
    # mask_tensor: H x W
    h, w = mask_tensor.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_np = mask_tensor.detach().cpu().numpy()
    for rgb, idx in colormap.items():
        mask_rgb[(mask_np == idx)] = rgb
    return mask_rgb

def save_batch_images(images, masks, epoch, batch_idx, mean, std, colormap, save_dir="debug_after_transform"):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = images.size(0)
    for i in range(batch_size):
        base_name = f"epoch{epoch}_batch{batch_idx}_img{i}"
        img_path = os.path.join(save_dir, f"{base_name}_image.png")
        mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
        
        save_image_tensor(images[i], img_path, mean, std)
        
        mask_rgb = decode_mask_tensor(masks[i], colormap)
        mask_bgr = cv.cvtColor(mask_rgb, cv.COLOR_RGB2BGR)
        cv.imwrite(mask_path, mask_bgr)

#Função Carregamento do melhor peso .pth
def load_best_model(model, best_model_path, device):
    if os.path.exists(best_model_path):
        print(f"Carregando melhor modelo salvo: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device)) #Carrega o melhor modelo
    else:
        print(f"Modelo {best_model_path} não encontrado!")
    return model

#Função que checa os pesos
def check_weight_compatibility(model, state_dict):
    model_dict = model.state_dict()
    matched_keys = []
    unmatched_keys = []
   
    for k in state_dict.keys():
        if k in model_dict and model_dict[k].shape == state_dict[k].shape:
            matched_keys.append(k)
        else:
            unmatched_keys.append(k)
   
    print(f"Total de pesos no modelo pré-treinado: {len(state_dict.keys())}")
    print(f"Pesos compatíveis carregados: {len(matched_keys)}")
    print(f"Pesos incompatíveis: {len(unmatched_keys)}\n")
 
    if unmatched_keys:
        print("\n❌ Pesos incompatíveis (não foram carregados):")
        print(unmatched_keys)

#Função Treino
def train(model, optimizer, criterion, epochs, train_loader, val_loader, num_classes, model_path, metrics, device):
    print("Iniciando treinamento...")    

    best_val_jaccard = 0.0
    best_model_jaccard = None    

    model.to(device) #Carrega o HRNET na GPU

    # Transformações com Kornia
    transform_train = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.8),
        K.RandomVerticalFlip(p=0.8),
        K.RandomAffine(
            degrees=15.0,          # rotação pequena
            scale=(1.05, 1.15),    # zoom para evitar bordas pretas
            p=0.8,
            align_corners=True
        ),
        data_keys=["input", "mask"],
        same_on_batch=True,
    ).to(device)

    #Loop de Treino
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0      #Loss começa com 0
        running_accuracy = 0.0  #Accuracy começa com 0
        running_jaccard = 0.0   #Jaccard começa com 0

        jaccard_classes_epoch_sum = np.zeros(num_classes)   #Vetor de Jaccard para cada classe

        #Loop dos Batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            
            images = images.to(device)  # Move as imagens para a GPU
            labels = labels.to(device).long()  # Move as máscaras para a GPU

            # Aplica transformações de data augmentation (imagem + máscara)
            images, labels = transform_train(images, labels)

            labels = labels.to(device).long().squeeze(1)
           
            #Treinamento
            outputs = model(images)            

            #Acesse apenas os logits
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            #Calcule a perda usando os logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()   #Zera o gradiente
            loss.backward()         #Backpropagation
            optimizer.step()        #Step optmizer
            
            #Atualizar o running_loss
            running_loss += loss.item()
            
            #Calcular acurácia e IoU
            accuracy = metrics.calculate_accuracy(outputs, labels)
            jaccard_metric, jaccard_classes = metrics.calculate_jaccard_sklearn(outputs, labels)   
            jaccard_classes_epoch_sum += np.array(jaccard_classes)
            running_accuracy += accuracy
            running_jaccard += jaccard_metric
            
            #Adicionar progresso durante a época
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / (batch_idx+1):.4f}, Accuracy: {running_accuracy / (batch_idx+1):.4f}, Jaccard: {running_jaccard / (batch_idx+1):.4f}")

        #Salvar as métricas por época
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)        
        epoch_jaccard = running_jaccard / len(train_loader)
        jaccard_classes_epoch_avg = jaccard_classes_epoch_sum / len(train_loader)    

        #Log para TensorBoard por época
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
        writer.add_scalar('Jaccard/train', epoch_jaccard, epoch)

        #Log do Jaccard por classe no TensorBoard
        for cls in range(num_classes):
            writer.add_scalar(f'Jaccard/train/class_{cls}', jaccard_classes_epoch_avg[cls], epoch)          

        #Validação
        model.eval()

        val_loss = 0.0                                          #Loss de validação da época
        val_accuracy = 0.0                                      #Acurácia de validação da época      
        val_jaccard = 0.0                                       #Jaccard de validação da época  
        val_jaccard_classes_epoch_sum = np.zeros(num_classes)   #Vetor de Jaccard das classes

        #Loop sobre o conjunto de validação
        with torch.no_grad():
            for images, labels in val_loader:   

                images = images.to(device) #Move as imagens de validação para a GPU
                labels = labels.to(device) #Move os labels de validação para a GPU

                #Validação
                outputs = model(images)

                #Extraia os logits
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                #Use os logits para calcular a perda
                loss = criterion(logits, labels)

                #Cálculo das métricas para cada batch
                val_loss += loss.item()
                accuracy = metrics.calculate_accuracy(outputs, labels)
                jaccard_metric, jaccard_classes = metrics.calculate_jaccard_sklearn(outputs, labels)
                val_jaccard_classes_epoch_sum += np.array(jaccard_classes)
                val_accuracy += accuracy
                val_jaccard += jaccard_metric

        #Cálculo das métricas por época
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_jaccard /= len(val_loader)
        val_jaccard_classes_epoch_avg = val_jaccard_classes_epoch_sum / len(val_loader)

        #Log para TensorBoard por época
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Jaccard/Validation', val_jaccard, epoch) 

        #Log da LR por época
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate_per_epoch', current_lr, epoch)

        #Log do Jaccard na validação no tensorBoard
        for cls in range(num_classes):
            writer.add_scalar(f'Jaccard/Validation/class_{cls}', val_jaccard_classes_epoch_avg[cls], epoch)                

        #Logs de treinamento e validação
        log_message = (
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_accuracy:.4f}, Train Jaccard: {epoch_jaccard:.4f},"
            f"Validation Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Val Jaccard {val_jaccard:.4f}\n"
        )
        
        #Print para console
        print(log_message.strip())

        #Salvar no arquivo
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message)
        
        #Se o melhor modelo
        if val_jaccard > best_val_jaccard:
            best_val_jaccard = val_jaccard
            best_model_jaccard = f"best_model_jaccard_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_jaccard)
            print(f"Melhor modelo jaccard salvo como '{best_model_jaccard}'.")        

        #Salvar sempre o último modelo treinado        
        torch.save(model.state_dict(), model_path)
        print(f"Último modelo salvo como " + model_path)
        
        #Step no scheduler
        scheduler.step(val_loss)
        #scheduler.step()

    #Retorna o modelo com melhor Jaccard
    print("Treinamento concluído.")
    return best_model_jaccard

#Função de Validação
def validate(model, criterion, val_loader, arquivo, metrics, device, n_classes):
    
    #Validação ajustada
    val_loss = 0.0
    val_accuracy = np.zeros(n_classes)
    val_f1_score = np.zeros(n_classes)
    val_jaccard = np.zeros(n_classes)
    accuracy_counts = np.zeros(n_classes)
    jaccard_counts = np.zeros(n_classes)   
    val_precision = np.zeros(n_classes)  
    val_recall = np.zeros(n_classes)     
    model.eval()       

    #Loop sobre o conjunto de validação
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validando"):
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, labels) #Cálculo da loss

            val_loss += loss.item() #Loss da validação
            
            #Cálculo de métricas por classe
            class_acc, acc_counts = metrics.calculate_class_accuracy(outputs, labels)
            mean_jaccard, class_jaccard = metrics.calculate_jaccard_sklearn(outputs, labels)
            
            val_accuracy += class_acc
            accuracy_counts += acc_counts
            val_jaccard += class_jaccard
            jaccard_counts += 1  #Contagem do número de batches para média            

            #Obter as saídas preditas para cálculos adicionais
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            class_precision, class_recall = metrics.calculate_precision_recall(preds, labels_np)
            val_precision += class_precision
            val_recall += class_recall

            #Cálculo do F1 Score
            class_f1_scores = metrics.calculate_f1_score(preds, labels_np)
            val_f1_score += class_f1_scores            

    #Calcular médias ponderadas
    val_loss /= len(val_loader)
    val_accuracy /= np.maximum(accuracy_counts, 1)
    val_f1_score /= np.maximum(accuracy_counts, 1)
    val_jaccard /= np.maximum(jaccard_counts, 1)    

    val_precision /= np.maximum(accuracy_counts, 1)
    val_recall /= np.maximum(accuracy_counts, 1)

    #Salvar métricas
    metrics_path = arquivo
    with open(metrics_path, "w") as f:
        f.write(f"Val Loss: {val_loss:.4f}\n")
        for cls in range(n_classes):
            f.write(f"Class {cls} - Accuracy: {val_accuracy[cls]:.4f}, "
                f"F1 Score: {val_f1_score[cls]:.4f}, Jaccard: {val_jaccard[cls]:.4f}, "
                f"Precision: {val_precision[cls]:.4f}, Recall: {val_recall[cls]:.4f}\n")

    print(f"Métricas por classe salvas em {metrics_path}")

#Função de Predict
def predict(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.33996263, 0.3619227, 0.33613345), std=(0.13796541, 0.13645774, 0.14186889))
    ])

    print("Iniciando predição e conversão nas imagens de validação...")
    image_files = sorted([f for f in os.listdir(img_dir_val) if os.path.isfile(os.path.join(img_dir_val, f))])

    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processando imagens"):
            image_path = os.path.join(img_dir_val, image_file)

            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            input_tensor = transform(image).unsqueeze(0).to(device)

            output = model(input_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            _, pred_mask = torch.max(logits, dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            # Converter a máscara de classes para RGB
            rgb_mask = convert_mask_to_rgb(pred_mask, CUSTOM_COLORMAP)

            # Salvar imagem RGB da máscara
            output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_prediction_rgb.png")
            cv.imwrite(output_path, cv.cvtColor(rgb_mask, cv.COLOR_RGB2BGR))  # OpenCV usa BGR

    print(f"Predições e conversões salvas em {output_dir}")

#Melhor modelo jaccard começando com 0
best_model_jaccard = None

#Setando GPUS
gpu_ids = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

#Liberação memória da GPU
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

#Setando hiperparâmetros
model_name = "HRNet"
learning_rate = 0.00005
epochs_value = 100
batch_size_value = 12
loss_ = "CrossEntropyDice"
size_crop = 512

#Timestamp para evitar conflitos
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

#Nome do log com hiperparâmetros
log_dir = f"./logs/{model_name}_lr={learning_rate}_Loss={loss_}_ep={epochs_value}_imageSize={size_crop}_bs={batch_size_value}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

#Caminho dataset
img_dir = '/homeLocal/lucas-alves/Dataset_Potsdam_Treino/train/images/images_crop_512'    #Imagem Treino
mask_dir = '/homeLocal/lucas-alves/Dataset_Potsdam_Treino/train/labels/labels_crop_512'   #Label Treino
img_dir_val = '/homeLocal/lucas-alves/Dataset_Potsdam_Treino/val/images/images_crop_512'  #Imagem validação
mask_dir_val = '/homeLocal/lucas-alves/Dataset_Potsdam_Treino/val/labels/labels_crop_512' #Label validação

CUSTOM_COLORMAP = {
    (255, 255, 255): 0,  # Impervious Surfaces (Branco)
    (0, 0, 255): 1,      # Building (Azul)
    (0, 255, 255): 2,    # Low Vegetation (ciano)
    (0, 255, 0): 3,      # Tree (Verde)
    (255, 255, 0): 4,    # Car (Amarelo)
    (255, 0, 0): 5       # Clutter/background (Vermelho)
}

#Métricas
metrics = SegmentationMetrics(num_classes=len(CUSTOM_COLORMAP))

#Imagens 512x512
#DataLoader de treino
print("Iniciando Dataset de treino...")
train_dataset = CustomDataset(
    img_dir, mask_dir, CUSTOM_COLORMAP,
    crop_size=(480, 480), final_size=(128, 128),
    mean=(0.33996263, 0.3619227, 0.33613345),
    std=(0.13796541, 0.13645774, 0.14186889),
    debug_mode=False, 
)
train_loader = DataLoader(train_dataset, batch_size=batch_size_value, shuffle=True)

#Imagens 512x512
#Dataloader de Validação
print("Iniciando Dataset de validação...")
val_dataset = CustomDataset(
    img_dir_val, mask_dir_val, CUSTOM_COLORMAP,
    crop_size=(480, 480), final_size=(128, 128),
    mean=(0.33996263, 0.3619227, 0.33613345),
    std=(0.13796541, 0.13645774, 0.14186889),
)
val_loader = DataLoader(val_dataset, batch_size=batch_size_value, shuffle=False)

# Modelo HRNet e parametôs para arquitetura
print("Iniciando modelo HRNet...")
in_channels = 3                     #Número de canais: RGB(3)
n_classes = len(CUSTOM_COLORMAP)    #Número de classes de saída
timestamps = 1                      #Timestamp
log_file_path = f"log_hrnet.txt"    #arquivo de escrita dos logs

#Certifica-se de que o arquivo é criado ou limpo antes do treinamento
with open(log_file_path, "w") as log_file:
    log_file.write("Treinamento iniciado\n")

# Caminho para os pesos pré-treinados
pretrained_weights_path = "pre_trained_pth_LARS.pth"

#Carrega o modelo HRNET
model = HighResolutionNet(in_channels=in_channels, n_classes=n_classes, timestamps=timestamps)

pretrained_weights_path = "pre_trained_pth_LARS.pth"
#pretrained_weights_path = "weights/best_model_jaccard_v14_17mapas_LARS_dataaugmentation_11.pth"
incompatibles = ['cls_head.weight', 'cls_head.bias', 'aux_head.3.weight', 'aux_head.3.bias']
 
if os.path.exists(pretrained_weights_path):
    print(f"Carregando pesos pré-treinados de {pretrained_weights_path}...")
    pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
   
    # Remove "module." e "model." caso existam nos pesos
    pretrained_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in pretrained_dict.items()}
 
    # Obtém os pesos atuais do modelo
    model_dict = model.state_dict()
    check_weight_compatibility(model, pretrained_dict)
    # Filtra apenas os pesos compatíveis (mesmas chaves e shapes)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
 
    # Atualiza os pesos do modelo sem carregar pesos incompatíveis
    model_dict.update(pretrained_dict)
   
    model.load_state_dict(model_dict, strict=False)    
 
    # Congelar todos os pesos do modelo
    for param in model.parameters():
        param.requires_grad = False
 
    for param in model.parameters():
        param.requires_grad = True    
 
    print("✅ Pesos pré-treinados carregados com sucesso.")
else:
    print("Nenhum peso pré-treinado encontrado. Treinando do zero.")

# GPUs específicas
if torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

#Imagens512x512
class_frequencies = {
    0: 0.2729,
    1: 0.2530,
    2: 0.2402,
    3: 0.1716,
    4: 0.0167,
    5: 0.0456,
}

#Calcula os pesos com base na frequência relativa
class_weights = {cls: 1 / math.log(freq + 1.1) for cls, freq in class_frequencies.items()}

#Normaliza os pesos pelo maior valor encontrado
max_weight = max(class_weights.values())
normalized_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

#Converte para tensor
weights = torch.tensor(list(normalized_weights.values()), dtype=torch.float).to(device)

#Imprime os pesos
print("Pesos normalizados:", weights)

#Loss
criterion = Dice_CrossEntropy_Loss(weight=weights, dice_weight=0.5, ce_weight=0.5).to(device)
#criterion = nn.CrossEntropyLoss()

#Optimizer e Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) 
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
# scheduler = PolynomialLR(optimizer, total_iters=14400, power=0.9)

#Iniciar treinamento
model_path = f"last_model_hrnet.pth"
best_model_jaccard = train(
     model, optimizer, criterion, epochs=epochs_value, train_loader=train_loader, 
     val_loader=val_loader, num_classes=len(CUSTOM_COLORMAP), model_path=model_path,
     metrics=metrics, device=device
)

if best_model_jaccard and os.path.exists(best_model_jaccard):
    model = load_best_model(model, best_model_jaccard, device)
    validate(model, criterion, val_loader, f"validation_best_jaccard.txt", metrics,device,len(CUSTOM_COLORMAP))  
    predict(model, f"/homeLocal/lucas-alves/predict_Potsdam_HRNetOCR") 
else:
    print("Nenhum modelo foi salvo com base no melhor jaccard.")   