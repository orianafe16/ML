"""Unsupervised analysis of the mutation profiles: PCA, MDS / t-SNE with Jaccard distance,
k-means and hierarchical clustering, and comparison of the clusters with the grade.

The input is the 0/1 matrix of the 20 genes. The grade is never used to build the clusters;
it is only used afterwards to describe them.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def fit_pca(X, scale):
    """PCA after centring (always) and, if `scale`, dividing each gene by its standard deviation.

    Scaling gives every gene unit variance, so rare mutations weigh as much as frequent ones;
    without scaling, the variance of a 0/1 gene is p(1 - p) and frequent mutations dominate.
    """
    Z = StandardScaler(with_std=scale).fit_transform(X)
    pca = PCA(random_state=config.SEED).fit(Z)
    return pca, pca.transform(Z)


def pca_variance_table(pca_centred, pca_scaled, n=10):
    return pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(n)],
        "Explained (centred)": pca_centred.explained_variance_ratio_[:n],
        "Cumulative (centred)": np.cumsum(pca_centred.explained_variance_ratio_)[:n],
        "Explained (scaled)": pca_scaled.explained_variance_ratio_[:n],
        "Cumulative (scaled)": np.cumsum(pca_scaled.explained_variance_ratio_)[:n],
    })


def n_components_for(pca, threshold):
    """Smallest number of components whose cumulative explained variance reaches `threshold`."""
    return int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), threshold) + 1)


def loadings_table(pca, features, n_components=2):
    return pd.DataFrame(pca.components_[:n_components].T, index=features,
                        columns=[f"PC{i + 1}" for i in range(n_components)])


# ---------------------------------------------------------------------------
# Distances and embeddings
# ---------------------------------------------------------------------------
def jaccard_distance(X):
    """Jaccard distance between mutation profiles: 1 - |A ∩ B| / |A ∪ B|.

    It ignores genes that are not mutated in either patient (shared absences), which are the
    vast majority. Two patients without any mutation get distance 0 (scipy convention), and a
    patient without mutations is at distance 1 from every patient with at least one.
    """
    return squareform(pdist(np.asarray(X).astype(bool), metric="jaccard"))


def mds_embedding(D, n_init=4):
    """Metric MDS in 2-D of a precomputed distance matrix. Returns coordinates and Kruskal stress-1."""
    mds = MDS(n_components=2, dissimilarity="precomputed", n_init=n_init, max_iter=500,
              random_state=config.SEED, normalized_stress=False)
    coords = mds.fit_transform(D)
    d_emb = squareform(pdist(coords))
    iu = np.triu_indices_from(D, k=1)
    stress1 = np.sqrt(((D[iu] - d_emb[iu]) ** 2).sum() / (D[iu] ** 2).sum())
    return coords, stress1


def tsne_embedding(D, perplexity=30):
    """t-SNE in 2-D of a precomputed distance matrix (non-linear, for visualisation only)."""
    tsne = TSNE(n_components=2, metric="precomputed", init="random", perplexity=perplexity,
                random_state=config.SEED)
    return tsne.fit_transform(D)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def kmeans_labels(X, k, n_init=50):
    """k-means with many random initialisations; the run with the lowest WCV is kept.

    On 0/1 data the squared Euclidean distance equals the number of genes in which two patients
    differ (Hamming distance), so k-means minimises within-cluster Hamming dispersion.
    """
    return KMeans(n_clusters=k, n_init=n_init, random_state=config.SEED).fit(X)


def ward_linkage(X):
    """Agglomerative clustering with Ward linkage (Euclidean distance on the 0/1 profiles)."""
    return linkage(np.asarray(X, dtype=float), method="ward")


def cluster_selection_table(X, ks=range(2, 9)):
    """For each k: within-cluster variation (k-means inertia), silhouette of k-means and Ward,
    and agreement (ARI) between k-means and Ward."""
    L = ward_linkage(X)
    rows = []
    for k in ks:
        km = kmeans_labels(X, k)
        ward = fcluster(L, k, criterion="maxclust")
        rows.append({"k": k, "WCV (k-means inertia)": km.inertia_,
                     "Silhouette k-means": silhouette_score(X, km.labels_),
                     "Silhouette Ward": silhouette_score(X, ward),
                     "ARI k-means vs Ward": adjusted_rand_score(km.labels_, ward)})
    return pd.DataFrame(rows)


def linkage_comparison(X, D_jaccard, k):
    """Cophenetic correlation and cluster sizes for several distance / linkage combinations."""
    D_jac = squareform(D_jaccard, checks=False)
    D_euc = pdist(np.asarray(X, dtype=float))
    rows = []
    for distance, method, D in (("Jaccard", "average", D_jac), ("Jaccard", "complete", D_jac),
                                ("Jaccard", "single", D_jac), ("Euclidean", "ward", D_euc)):
        L = linkage(D, method=method)
        labels = fcluster(L, k, criterion="maxclust")
        sizes = np.bincount(labels)[1:]
        rows.append({"Distance": distance, "Linkage": method,
                     "Cophenetic correlation": cophenet(L, D)[0],
                     "Clusters obtained": len(sizes),
                     "Cluster sizes": ", ".join(str(s) for s in sorted(sizes, reverse=True)),
                     "Largest cluster (%)": 100 * sizes.max() / sizes.sum()})
    return pd.DataFrame(rows)


def kmeans_stability(X, k, best_labels, n_runs=20):
    """ARI between the best k-means solution (n_init=50) and single-initialisation runs."""
    return np.array([adjusted_rand_score(best_labels,
                                         KMeans(n_clusters=k, n_init=1, random_state=s).fit(X).labels_)
                     for s in range(n_runs)])


def order_clusters(labels, y):
    """Rename clusters C1, C2, ... in decreasing proportion of GBM (deterministic names)."""
    labels = np.asarray(labels)
    order = pd.Series(y).groupby(labels).mean().sort_values(ascending=False).index
    mapping = {old: f"C{i + 1}" for i, old in enumerate(order)}
    return np.array([mapping[l] for l in labels])


def agreement_with_grade(labels, y):
    return {"ARI": adjusted_rand_score(y, labels), "NMI": normalized_mutual_info_score(y, labels)}


def cluster_profiles(df, labels, genes):
    """Size, % GBM, median age and mutation prevalence of each cluster."""
    g = df.assign(Cluster=labels).groupby("Cluster")
    prof = pd.DataFrame({"n": g.size(), "GBM (%)": 100 * g["Grade"].mean(),
                         "Median age": g["Age_at_diagnosis"].median()})
    prevalence = 100 * g[genes].mean()
    return prof, prevalence
