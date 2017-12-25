from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def viz_pca(train, labels):
    pca = PCA(n_components=2)
    pca.fit(train)
    xxx = pca.transform(train)

    plt.scatter(xxx[:,0], xxx[:,1], c=labels, cmap=plt.cm.spectral)
    plt.xlabel('PCA_1')
    plt.ylabel('PCA_2')
    plt.title('PCA')
    plt.savefig('../results/pca.png')
    plt.show()

def viz_tsne(train, labels):
    tsne = TSNE()
    xxx = tsne.fit_transform(np.array(train))

    plt.scatter(xxx[:,0], xxx[:,1], c=labels, cmap=plt.cm.spectral)
    plt.xlabel('TSNE_1')
    plt.ylabel('TSNE_2')
    plt.title('T-SNE')
    plt.savefig('../results/tsne.png')
    plt.show()  