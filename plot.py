import seaborn
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

__all__ = ('plot_heat_map', 'plot_history')

# confusion matrix
def plot_heat_map(y_test, y_pred, path, model_name, show=False, labels=None):
    """Plot and save a confusion matrix heatmap.

    Args:
        y_test: array-like of true labels (int or str)
        y_pred: array-like of predicted labels (int or str)
        path: directory to save the figure
        model_name: base name for saved file
        show: whether to call plt.show()
        labels: optional list of label names to use as ticks (ordered)
    """
    con_mat = confusion_matrix(y_test, y_pred)
    # normalize
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # plot
    plt.figure(figsize=(8, 8))
    if labels is not None:
        seaborn.heatmap(con_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        seaborn.heatmap(con_mat, annot=True, fmt='d', cmap='Blues')
    # plt.xlim(0, con_mat.shape[1])
    # plt.ylim(0, con_mat.shape[0])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(path, model_name + '_confusion_matrix.png'))
    if show:
        plt.show()

def plot_history(history, path, model_name, show=False):
    plt.figure(figsize=(8, 8))
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, model_name + '_accuracy.png'))
    if show:
        plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, model_name + '_loss.png'))
    if show:
        plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_lr'])
    plt.title('Learning Rate')
    plt.ylabel('lr')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, model_name + '_lr.png'))
    if show:
        plt.show()