import numpy
# matplotlib.use("TkAgg")
import umap
from matplotlib import pyplot
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # _x = numpy.load("/Volumes/bstorage/home/chexov/smoke_attn/i3d/BX137_SRNA_03.mov_vecs.npy")

    _x = numpy.load("/Volumes/bstorage/home/chexov/smoke_attn/i3d/BX137_SRNA_02.mov_vecs.npy")
    _x = numpy.reshape(_x, (194, -1))
    print(_x.shape)

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

    _ex = embedding[:, 0]
    _ey = embedding[:, 1]
    _ez = embedding[:, 2]

    with open("/tmp/out.tsv", "w") as _f:
        for i in range(0, embedding.shape[0]):
            a, b, c = embedding[i]
            s = "%s\t%s\t%s\n" % (a, b, c)
            _f.write(s)
            print(s)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(_ex, _ey, _ez, alpha=0.8, cmap="Accent")
    # , edgecolors='none', s=30, label=labels)

    pyplot.show()
