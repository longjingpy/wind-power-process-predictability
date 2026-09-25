"""Training-only representations and clustering with explicit out-of-sample use.

GASF: Wang and Oates (2015), arXiv:1506.00327, Gramian angular fields section.
For signed normalized paths G(x)=xx^T-cc^T, c=sqrt(1-x^2). This encoding
identifies x only up to global sign; the sign-bit experiment isolates that
loss from the effect of giving a network the complete original sequence.
CLARA-style subsampling: Kaufman and Rousseeuw (1990), Finding Groups in Data,
doi:10.1002/9780470316801, clustering large applications chapter. Fixed sample
budgets and full-training-set objectives make medoid fitting scalable.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURES = ["duration_hours", "amplitude", "power_range", "total_variation",
            "max_abs_rate_per_hour", "max_abs_chord_residual", "curvature_l1", "pre_mean", "post_mean"]


def gasf(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or (np.abs(x) > 1+1e-6).any():
        raise ValueError("Finite signed unit-domain shapes required")
    x = np.clip(x, -1, 1)
    c = np.sqrt(np.maximum(0, 1-x*x))
    return x[:, :, None]*x[:, None, :]-c[:, :, None]*c[:, None, :]


def polarity_bit(x):
    x = np.asarray(x)
    anchor = np.abs(x).argmax(axis=1)
    return np.sign(x[np.arange(len(x)), anchor])[:, None]


def invert_gasf(g, sign_bit):
    """Exact inversion when a nonzero anchor's sign is supplied."""
    g = np.asarray(g, float)
    magnitude = np.sqrt(np.maximum(0, (np.diagonal(g, axis1=1, axis2=2)+1)/2))
    c = np.sqrt(np.maximum(0, 1-magnitude*magnitude))
    outer = g+c[:, :, None]*c[:, None, :]
    anchor = magnitude.argmax(axis=1)
    denominator = magnitude[np.arange(len(g)), anchor]*np.asarray(sign_bit).reshape(-1)
    if (np.abs(denominator) < 1e-12).any():
        raise ValueError("Nonzero reference sign required")
    return outer[np.arange(len(g)), :, anchor]/denominator[:, None]


def raw_encoding(x, name):
    if name in ("raw25", "raw_pca6", "statistics9"):
        return np.asarray(x, float)
    encoded = gasf(x).reshape(len(x), -1)
    if name == "gaf_signed_pca6":
        return np.c_[encoded, x]
    if name in ("gaf_pca6", "gaf_bit6"):
        return encoded
    raise ValueError(f"Unknown representation {name}")


class Representation:
    def __init__(self, name):
        self.name = name

    def fit(self, x):
        encoded = raw_encoding(x, self.name)
        self.scaler = StandardScaler().fit(encoded)
        transformed = self.scaler.transform(encoded)
        dimensions = 5 if self.name == "gaf_bit6" else 6
        self.pca = PCA(dimensions, svd_solver="full").fit(transformed) if self.name.endswith("pca6") or self.name == "gaf_bit6" else None
        return self

    def transform(self, x):
        result = self.scaler.transform(raw_encoding(x, self.name))
        if self.pca is not None:
            result = self.pca.transform(result)
        if self.name == "gaf_bit6":
            result = np.c_[result, polarity_bit(x)]
        return result


class SampledMedoids:
    def __init__(self, n_clusters, random_state, sample_size=512, repetitions=10, iterations=20):
        self.k, self.seed, self.sample_size = n_clusters, random_state, sample_size
        self.repetitions, self.iterations = repetitions, iterations

    def fit(self, x):
        x = np.asarray(x, float)
        if len(x) < self.k:
            raise ValueError("Training set smaller than cluster count")
        rng = np.random.default_rng(self.seed)
        best = np.inf
        for _ in range(self.repetitions):
            indices = rng.choice(len(x), min(self.sample_size, len(x)), replace=False)
            sample = x[indices]
            distances = cdist(sample, sample)
            medoids = rng.choice(len(sample), self.k, replace=False)
            for _ in range(self.iterations):
                labels = distances[:, medoids].argmin(axis=1)
                updated = medoids.copy()
                for cluster in range(self.k):
                    members = np.flatnonzero(labels == cluster)
                    if len(members):
                        updated[cluster] = members[distances[np.ix_(members, members)].sum(axis=1).argmin()]
                if np.array_equal(updated, medoids):
                    break
                medoids = updated
            centers = sample[medoids]
            objective = float(cdist(x, centers).min(axis=1).mean())
            if objective < best:
                best, self.cluster_centers_ = objective, centers.copy()
                self.medoid_indices_ = indices[medoids]
        self.objective_ = best
        return self

    def predict(self, x):
        return cdist(x, self.cluster_centers_).argmin(axis=1)


def fit_cluster(x, algorithm, k, seed):
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
    if algorithm == "gmm":
        return GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                               max_iter=300, reg_covar=1e-5, random_state=seed).fit(x)
    if algorithm == "medoids":
        return SampledMedoids(k, seed).fit(x)
    raise ValueError(f"Unknown clustering algorithm {algorithm}")
