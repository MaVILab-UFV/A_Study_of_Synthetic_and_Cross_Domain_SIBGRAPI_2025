import numpy as np
import torch
from sklearn.metrics import jaccard_score

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def calculate_precision_recall(self, predictions, labels):
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            tp = np.sum((predictions == cls) & (labels == cls))
            fp = np.sum((predictions == cls) & (labels != cls))
            fn = np.sum((predictions != cls) & (labels == cls))

            precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return precision, recall

    def calculate_accuracy(self, outputs, labels):
        logits = outputs[0] if isinstance(outputs, tuple) else outputs  
        _, preds = torch.max(logits, dim=1)  

        correct = (preds == labels).sum().item()
        total = labels.numel()  
        accuracy = correct / total 
        return accuracy

    def calculate_jaccard_sklearn(self, outputs, labels):
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        _, preds = torch.max(logits, dim=1)

        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        jaccard_per_class = []

        for cls in range(self.num_classes):
            y_true = (labels_np == cls).astype(int)
            y_pred = (preds_np == cls).astype(int)

            jaccard = jaccard_score(y_true, y_pred, average='binary', zero_division=0)
            jaccard_per_class.append(jaccard)

        mean_jaccard = np.mean(jaccard_per_class)
        return mean_jaccard, jaccard_per_class

    def calculate_f1_score(self, predictions, labels):
        f1_scores = np.zeros(self.num_classes)
        
        for cls in range(self.num_classes):
            tp = np.sum((predictions == cls) & (labels == cls))
            fp = np.sum((predictions == cls) & (labels != cls))
            fn = np.sum((predictions != cls) & (labels == cls))

            if tp + fp + fn > 0:
                f1_scores[cls] = 2 * tp / (2 * tp + fp + fn)
            else:
                f1_scores[cls] = 0.0

        return f1_scores

    def calculate_class_accuracy(self, outputs, labels):
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        _, preds = torch.max(logits, dim=1)

        class_accuracy = np.zeros(self.num_classes)
        class_counts = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            pred_mask = (preds == cls)
            label_mask = (labels == cls)

            correct = (pred_mask & label_mask).sum().item()
            total = label_mask.sum().item()

            if total > 0:
                class_accuracy[cls] = correct / total
            class_counts[cls] += 1

        return class_accuracy, class_counts
