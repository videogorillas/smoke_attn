import os

import numpy
# matplotlib.use("TkAgg")
import umap
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def project(_x):
    global embedding
    emb = "umap"
    if emb == "pca3":
        pca = PCA(n_components=3)
        embedding = pca.fit_transform(_x)
        pyplot.title('PCA')

        print(embedding.shape,
              pca.explained_variance_ratio_)

    elif emb == "umap":
        embedding = umap.UMAP(n_neighbors=9,
                              n_components=3,
                              verbose=True,
                              n_epochs=100,
                              ).fit_transform(_x)
        pyplot.title('UMAP')
        # n_neighbors=5,
        #   verbose=True,
        #   min_dist=0.3,
        #   metric='correlation')
    else:
        embedding = []

    return embedding


def emb_to_tsv(_embedding, out_filename):
    key = os.path.basename(out_filename)
    with open(out_filename + "_meta.tsv", "w", ) as _mf:
        s = "fn%s\t%s\n" % ("Frame", "Filename")
        _mf.write(s)

        with open(out_filename, "w", ) as _f:
            for i in range(0, _embedding.shape[0]):
                a, b, c = _embedding[i]
                s = "%s\t%s\t%s\n" % (a, b, c)
                _f.write(s)

                s = "fn%s\t%s\n" % (i, key)
                _mf.write(s)


def plot_emb(_embedding, ax):
    _ex = _embedding[:, 0]
    _ey = _embedding[:, 1]
    _ez = _embedding[:, 2]
    ax.scatter(_ex, _ey, _ez, alpha=0.8, cmap="Accent")


if __name__ == '__main__':
    # _x = numpy.load("/Volumes/bstorage/home/chexov/smoke_attn/i3d/BX137_SRNA_01.npy")
    fig = pyplot.figure()
    ax = Axes3D(fig)

    _x = numpy.load("/Volumes/bstorage/home/chexov/macg_embedding/BX137_SRNA_01.mov.npy")
    embedding = project(_x)
    plot_emb(embedding, ax)
    emb_to_tsv(embedding, "/tmp/umap_BX137_SRNA_01.mov.tsv")

    _x = numpy.load("/Volumes/bstorage/home/chexov/macg_embedding/BX137_SRNA_03.mov.npy")
    embedding = project(_x)
    plot_emb(embedding, ax)
    emb_to_tsv(embedding, "/tmp/umap_BX137_SRNA_03.mov.tsv")

    # kmeans = KMeans(n_clusters=4, random_state=0).fit(embedding)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)
    # 
    # fig = pyplot.figure()
    # ax = fig.add_subplot(212)
    # k_means_labels = pairwise_distances_argmin(embedding, kmeans.cluster_centers_)
    # for k in range(0, 4):
    #     my_members = k_means_labels == k
    #     ax.scatter(_ex[my_members], _ey[my_members], alpha=0.8, cmap="Accent")
    # 
    # ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=4200, alpha=0.3, cmap="spring")

    pyplot.show()
