import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix (%)")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 14}

    matplotlib.rc('font', **font)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

a = [[0.12, 0.12, 0., 0., 0.25, 0., 0.12, 0.38, 0.],
 [0., 0.5, 0.5, 0., 0., 0., 0., 0., 0.],
 [0., 0.06, 0.88, 0., 0.06, 0., 0., 0., 0.],
 [0., 0., 0., 0.92, 0., 0.08, 0., 0., 0.],
 [0., 0., 0., 0., 0.75, 0.25, 0., 0., 0.],
 [0., 0., 0., 0., 0.33, 0.67, 0., 0., 0.],
 [0., 0., 0.08, 0., 0.17, 0., 0.67, 0., 0.08],
 [0., 0.08, 0., 0.08, 0., 0.08, 0.,   0.75, 0.],
 [0., 0.17, 0.25, 0., 0., 0., 0., 0.08, 0.5]]

b = [[ 1, 1, 0, 0, 2, 0, 1, 3, 0],
 [ 0, 6, 6, 0, 0, 0, 0, 0, 0],
 [ 0, 1, 14, 0, 1, 0, 0, 0, 0],
 [ 0, 0, 0, 11, 0, 1, 0, 0, 0],
 [ 0, 0, 0, 0, 9, 3, 0, 0, 0],
 [ 0, 0, 0, 0, 4, 8, 0, 0, 0],
 [ 0, 0, 1, 0, 2, 0, 8, 0, 1],
 [ 0, 1, 0, 1, 0, 1, 0, 9, 0],
 [ 0, 2, 3, 0, 0, 0, 0, 1, 6]]