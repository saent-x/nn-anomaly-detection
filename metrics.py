import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import datetime


def plot_training_history(history, save_dir="training_plots"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-dark-palette')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png")
    plt.close()

    metrics_summary = {
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'best_val_accuracy': max(history.history['val_accuracy']),
        'best_val_loss': min(history.history['val_loss']),
        'epochs_trained': len(history.history['accuracy'])
    }

    with open(f"{save_dir}/metrics_summary.txt", 'w') as f:
        for metric, value in metrics_summary.items():
            f.write(f"{metric}: {value:.4f}\n")

    return metrics_summary


def evaluate_model_performance(model, test_ds, save_dir="evaluation_plots"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    y_pred_proba = model.predict(test_ds)
    y_pred_proba = np.squeeze(y_pred_proba)  # Ensure y_pred_proba is 1D
    y_pred = (y_pred_proba > 0.5).astype(int)

    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.squeeze(y_true)  # Ensure y_true is 1D

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{save_dir}/roc_curve.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()

    report = classification_report(y_true, y_pred)
    with open(f"{save_dir}/classification_report.txt", 'w') as f:
        f.write(report)

    return {
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }