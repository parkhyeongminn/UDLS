from sklearn.metrics import roc_auc_score, roc_curve
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc


def reliability_diagram(preds, labels, bins=10):
    bin_bounds = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_bounds[:-1]
    bin_uppers = bin_bounds[1:]
    
    bin_centers = []
    accuracies = []
    gaps =[]
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.where((preds > bin_lower) & (preds <= bin_upper))[0]
        if len(in_bin) > 0:
            bin_accuracy = labels[in_bin].mean()
            bin_center = (bin_lower+bin_upper)/2
            bin_centers.append(bin_center)
            accuracies.append(bin_accuracy)
            gaps.append(bin_center-bin_accuracy)
    
    plt.figure(figsize=(8, 8))
    plt.bar(bin_centers, accuracies, width=1.0/bins, align='center', edgecolor='black')
    # plt.plot(bin_centers, gaps, color='red', label='Gap', marker='o')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.title('Reliability Diagram')
    plt.show()



def expected_calibration_error(preds, labels, bins=10):
    bin_bounds = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_bounds[:-1]
    bin_uppers = bin_bounds[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.where((preds > bin_lower) & (preds <= bin_upper))[0]
        if len(in_bin) > 0:
            bin_accuracy = labels[in_bin].mean()
            bin_confidence = preds[in_bin].mean()
            bin_error = np.abs(bin_confidence - bin_accuracy)
            bin_weight = len(in_bin) / len(preds)
            
            ece += bin_error * bin_weight
    
    return ece
