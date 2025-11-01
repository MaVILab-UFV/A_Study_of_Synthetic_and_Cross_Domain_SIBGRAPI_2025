import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import shutil

# Mapeamento de cores para classes (ground truth)
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

DEFAULT_CLASS = -1

# Lista de classes para rótulos das matrizes
target_labels = ["Background", "OutdoorStructures ", "Buildings", "PGround", "NonPGround", "TrainTracks", "Plants", "W_Vehicles", "Water", "People"]

# Função para converter imagem colorida em matriz de classes
def convert_image_to_classes(image_path, colormap, default_class=DEFAULT_CLASS):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Carrega a imagem em BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST) #tirar isso
    class_map = np.full((img.shape[0], img.shape[1]), default_class, dtype=np.int32)
    
    for rgb, class_id in colormap.items():
        mask = np.all(img == np.array(rgb), axis=-1)  # Máscara para encontrar os pixels dessa cor
        class_map[mask] = class_id
    
    return class_map

# Função para salvar matrizes de confusão em porcentagem
def save_confusion_matrix(conf_matrix, filename, title):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_pct = np.divide(conf_matrix, row_sums, where=row_sums != 0)  # Evita divisão por zero
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(conf_matrix_pct, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=target_labels, yticklabels=target_labels)
    plt.xlabel("Predito")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

# Função para salvar a máscara de erro e a imagem sobreposta
def save_error_mask_and_overlay(gt_path, pred_path, original_image_path, output_folder, image_name):
    if not os.path.exists(original_image_path):
        print(f"Erro: Imagem original não encontrada em {original_image_path}")
        return
    
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    original_image = cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_NEAREST) #tirar isso
    if original_image is None:
        print(f"Erro: Falha ao carregar a imagem {original_image_path}")
        return
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)    
    
    gt_classes = convert_image_to_classes(gt_path, CUSTOM_COLORMAP)
    pred_classes = convert_image_to_classes(pred_path, CUSTOM_COLORMAP)
    
    # Criando a máscara binária de erro (1 para erro, 0 para acerto)
    valid_mask = (gt_classes != DEFAULT_CLASS) & (pred_classes != DEFAULT_CLASS)
    
    error_mask = np.zeros_like(gt_classes, dtype=np.uint8)
    error_mask[valid_mask] = (gt_classes[valid_mask] != pred_classes[valid_mask]).astype(np.uint8) * 255
    

    cv2.imwrite(os.path.join(output_folder, f"{image_name}_error_mask.png"), error_mask)

    # Criando uma cópia da imagem original para sobreposição
    overlay = original_image.copy()

    # Criando uma imagem colorida da máscara para visualização
    error_colored = np.zeros_like(original_image)
    error_colored[:, :, 0] = error_mask  # Define a máscara como vermelha (canal R)


    #error_colored[:, :, 1] = error_mask  # Verde
    #error_colored[:, :, 2] = error_mask  # Azul

    # Aplicando sobreposição com transparência apenas nos pixels de erro
    alpha = 0.2  # Grau de transparência
    mask_indices = error_mask > 0  # Índices onde há erro
    overlay[mask_indices] = cv2.addWeighted(original_image[mask_indices], 1 - alpha, error_colored[mask_indices], alpha, 0)

    # Salvando a imagem final com a sobreposição
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_error_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Caminho das imagens
gt_folder = '/raid/dados/es111286/datasets/swiss_okutama/regions/Fold3/test/ground_truth'
pred_folder = "/raid/dados/es111286/repositorios/Projeto_Fazendas_Preparacao_Dataset_Japao/predict_new/hrnet_FOLD3_HRNet_V04_PesosPotsdam_CE_adam1e4_b36_img128_jaccard"
output_folder = '/raid/dados/es111286/repositorios/Projeto_Fazendas_Preparacao_Dataset_Japao/matrizes_confusao_new/hrnet_FOLD3_HRNet_V04_PesosPotsdam_CE_adam1e4_b36_img128_jaccard'
image_folder = "/raid/dados/es111286/datasets/swiss_okutama/regions/Fold3/test/images" 

os.makedirs(output_folder, exist_ok=True)

# Lista de imagens
image_files = [f for f in os.listdir(gt_folder) if f.endswith(".png")]

# Imagens específicas
specific_images = [
    #fold1
    # ("okutama_02_50_012_0,0", "okutama_02_50_012_0,0_prediction_rgb"),
    # ("okutama_02_50_013_0,0", "okutama_02_50_013_0,0_prediction_rgb"),
    # ("okutama_02_50_014_0,0", "okutama_02_50_014_0,0_prediction_rgb"),
    # ("okutama_02_50_015_0,0", "okutama_02_50_015_0,0_prediction_rgb"),
    # ("okutama_02_50_016_0,0", "okutama_02_50_016_0,0_prediction_rgb"),
    # ("okutama_02_50_018_0,0", "okutama_02_50_018_0,0_prediction_rgb"),
    # ("swiss_IMG_8709_0,0", "swiss_IMG_8709_0,0_prediction_rgb"),
    # ("swiss_IMG_8710_0,0", "swiss_IMG_8710_0,0_prediction_rgb"),
    # ("swiss_IMG_8711_0,0", "swiss_IMG_8711_0,0_prediction_rgb"),
    # ("swiss_IMG_8712_0,0", "swiss_IMG_8712_0,0_prediction_rgb"),
    # ("swiss_IMG_8722_0,0", "swiss_IMG_8722_0,0_prediction_rgb"),
    # ("swiss_IMG_8723_0,0", "swiss_IMG_8723_0,0_prediction_rgb"),
    # ("swiss_IMG_8724_0,1", "swiss_IMG_8724_0,1_prediction_rgb"),
    # ("swiss_IMG_8754_0,0", "swiss_IMG_8754_0,0_prediction_rgb"),

    #fold2
    # ("okutama_04_90_007_1,0", "okutama_04_90_007_1,0_prediction_rgb"),
    # ("okutama_04_90_009_0,0", "okutama_04_90_009_0,0_prediction_rgb"),
    # ("okutama_04_90_012_0,0", "okutama_04_90_012_0,0_prediction_rgb"),
    # ("okutama_04_90_013_0,0", "okutama_04_90_013_0,0_prediction_rgb"),
    # ("okutama_04_90_014_0,0", "okutama_04_90_014_0,0_prediction_rgb"),
    # ("okutama_04_90_015_0,0", "okutama_04_90_015_0,0_prediction_rgb"),
    # ("swiss_IMG_8736_0,0", "swiss_IMG_8736_0,0_prediction_rgb"),
    # ("swiss_IMG_8737_0,0", "swiss_IMG_8737_0,0_prediction_rgb"),
    # ("swiss_IMG_8738_0,0", "swiss_IMG_8738_0,0_prediction_rgb"),
    # ("swiss_IMG_8739_0,0", "swiss_IMG_8739_0,0_prediction_rgb"),
    # ("swiss_IMG_8740_0,0", "swiss_IMG_8740_0,0_prediction_rgb"),

    #fold3    
    ("okutama_02_50_027_0,0", "okutama_02_50_027_0,0_prediction_rgb"),
    ("okutama_02_50_028_0,0", "okutama_02_50_028_0,0_prediction_rgb"),
    ("okutama_02_50_031_0,0", "okutama_02_50_031_0,0_prediction_rgb"),
    ("okutama_02_50_035_0,0", "okutama_02_50_035_0,0_prediction_rgb"),
    ("okutama_hs_90_009_0,0", "okutama_hs_90_009_0,0_prediction_rgb"),
    ("okutama_hs_90_015_0,0", "okutama_hs_90_015_0,0_prediction_rgb"),
    ("swiss_IMG_8714_0,0", "swiss_IMG_8714_0,0_prediction_rgb"),
    ("swiss_IMG_8716_0,0", "swiss_IMG_8716_0,0_prediction_rgb"),
    ("swiss_IMG_8720_0,0", "swiss_IMG_8720_0,0_prediction_rgb"),
    ("swiss_IMG_8731_0,0", "swiss_IMG_8731_0,0_prediction_rgb"),
    ("swiss_IMG_8747_0,0", "swiss_IMG_8747_0,0_prediction_rgb"),
    ("swiss_IMG_8749_0,0", "swiss_IMG_8749_0,0_prediction_rgb"),

]


# Matriz de confusão total
total_conf_matrix = np.zeros((10, 10), dtype=int)

scores = []
matrices = {}

for image_file in image_files:
    gt_path = os.path.join(gt_folder, image_file)
    pred_path = os.path.join(pred_folder, image_file.replace(".png", "_prediction_rgb.png"))
    
    if not os.path.exists(pred_path):
        continue
    
    gt_classes = convert_image_to_classes(gt_path, CUSTOM_COLORMAP)
    pred_classes = convert_image_to_classes(pred_path, CUSTOM_COLORMAP)
    
    # Remover pixels desconhecidos antes da matriz de confusão
    valid_mask = (gt_classes != DEFAULT_CLASS) & (pred_classes != DEFAULT_CLASS)
    labels = list(range(10))
    conf_matrix = confusion_matrix(gt_classes[valid_mask].ravel(), pred_classes[valid_mask].ravel(), labels=labels)

    
    total_conf_matrix += conf_matrix
    
    score = np.trace(conf_matrix)
    scores.append((score, image_file, conf_matrix))
    matrices[image_file] = conf_matrix

# Salvar a matriz de confusão total
save_confusion_matrix(total_conf_matrix, "total_confusion_matrix.png", "Matriz de Confusão Total")

print("Matriz de confusão total gerada e salva com sucesso!")   

# Gerar matriz de confusão para imagens específicas e salvar erro e overlay
for gt_name, pred_name in specific_images:
    gt_path = os.path.join(gt_folder, gt_name + ".png")
    pred_path = os.path.join(pred_folder, pred_name + ".png")
    # Tenta encontrar a imagem original com extensão .png ou .jpg
    original_image_path_png = os.path.join(image_folder, gt_name + ".png")
    original_image_path_jpg = os.path.join(image_folder, gt_name + ".JPG")

    if os.path.exists(original_image_path_png):
        original_image_path = original_image_path_png
    elif os.path.exists(original_image_path_jpg):
        original_image_path = original_image_path_jpg
    else:
        print(f"Imagem original não encontrada para {gt_name} (.png ou .jpg)")
        continue

    
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        gt_classes = convert_image_to_classes(gt_path, CUSTOM_COLORMAP)
        pred_classes = convert_image_to_classes(pred_path, CUSTOM_COLORMAP)
        valid_mask = (gt_classes != DEFAULT_CLASS) & (pred_classes != DEFAULT_CLASS)
        labels = list(range(10))
        conf_matrix = confusion_matrix(gt_classes[valid_mask].ravel(), pred_classes[valid_mask].ravel(), labels=labels)

        save_confusion_matrix(conf_matrix, f"{gt_name}_confusion_matrix.png", f"Matriz - {gt_name}")
        
        shutil.copy(gt_path, os.path.join(output_folder, f"{gt_name}_ground_truth.png"))
        shutil.copy(pred_path, os.path.join(output_folder, f"{gt_name}_prediction.png"))
        
        # Salvar a máscara de erro e a imagem sobreposta
        save_error_mask_and_overlay(gt_path, pred_path, original_image_path, output_folder, gt_name)
        print(f"Matriz, máscara de erro e sobreposição para {gt_name} salvas.")

# ======= Cálculo de métricas =======
print("\n==== Métricas por classe ====")

TP = np.diag(total_conf_matrix)
FP = total_conf_matrix.sum(axis=0) - TP
FN = total_conf_matrix.sum(axis=1) - TP
TN = total_conf_matrix.sum() - (TP + FP + FN)

precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
iou = np.divide(TP, TP + FP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FP + FN) != 0)
f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)
accuracy = np.divide(TP + TN, total_conf_matrix.sum(), out=np.zeros_like(TP, dtype=float))

# for i, label in enumerate(target_labels):
#     print(f"Classe: {label} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | IoU: {iou[i]:.4f} | F1-score: {f1[i]:.4f} | Accuracy: {accuracy[i]:.4f}")

# print("==== Métricas médias ====")
# print(f"Mean Precision: {np.mean(precision):.4f} | Mean Recall: {np.mean(recall):.4f} | Mean IoU: {np.mean(iou):.4f} | Mean F1-score: {np.mean(f1):.4f} | Mean Accuracy: {np.mean(accuracy):.4f}")


# Cálculo de métricas por classe
precision_list = []
recall_list = []
iou_list = []
f1_list = []
accuracy_list = []

metrics_output_path = os.path.join(output_folder, "metricas_por_classe.txt")

with open(metrics_output_path, "w") as f:
    f.write("==== Métricas por classe ====\n")
    for i in range(len(target_labels)):
        TP = total_conf_matrix[i, i]
        FP = total_conf_matrix[:, i].sum() - TP
        FN = total_conf_matrix[i, :].sum() - TP
        TN = total_conf_matrix.sum() - (TP + FP + FN)

        print(f"\nClasse: {target_labels[i]}")
        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        #accuracy = (TP + TN) / total_conf_matrix.sum() if total_conf_matrix.sum() > 0 else 0.0
        
        if (TP + FP) > 0: #se não fez nenhum predict não calcula acurácia
            accuracy = (TP + TN) / (TP + FP + FN + TN)
        else:
            accuracy = 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        iou_list.append(iou)
        f1_list.append(f1_score)
        accuracy_list.append(accuracy)

        f.write(f"Classe: {target_labels[i]} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
                f"IoU: {iou:.4f} | F1-score: {f1_score:.4f} | Accuracy: {accuracy:.4f}\n")

    # Cálculo das médias
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_iou = np.mean(iou_list)
    mean_f1 = np.mean(f1_list)
    mean_acc = np.mean(accuracy_list)

    f.write("\n==== Métricas médias ====\n")
    f.write(f"Mean Precision: {mean_precision:.4f} | "
            f"Mean Recall: {mean_recall:.4f} | "
            f"Mean IoU: {mean_iou:.4f} | "
            f"Mean F1-score: {mean_f1:.4f} | "
            f"Mean Accuracy: {mean_acc:.4f}\n")

print(f"Métricas salvas em {metrics_output_path}")
