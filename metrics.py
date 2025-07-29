"""define the metrics"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import CLASS_NAMES

def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, title="Confusion Matrix"):
    """ plotting the confusion matrix """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp1.plot(cmap="Blues", values_format="d", ax=axs[0], colorbar=True)
    axs[0].set_title("Counts")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=class_names)
    disp2.plot(cmap="viridis", values_format=".1f", ax=axs[1], colorbar=True)
    axs[1].set_title("Percentage")

    plt.tight_layout()
    plt.show()


