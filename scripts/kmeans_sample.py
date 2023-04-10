import numpy as np
from time import time
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)
labels = digits.target

k = 10
samples, features = data.shape
sample_size = 300


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print(82 * '_')
    print(f"init: {name}")
    print(f"time: {(time() - t0):0.2f} sec")
    print(f"inertia: {estimator.inertia_:0.4f}")
    print(f"homogeneity_score: {metrics.homogeneity_score(labels, estimator.labels_):0.4f}")
    print(f"completeness_score: {metrics.completeness_score(labels, estimator.labels_):0.4f}")
    print(f"v_measure_score: {metrics.v_measure_score(labels, estimator.labels_):0.4f}")
    print(f"adjusted_rand_score: {metrics.adjusted_rand_score(labels, estimator.labels_):0.4f}")
    print(f"adjusted_mutual_info_score: {metrics.adjusted_mutual_info_score(labels,  estimator.labels_):0.4f}")
    print(f"silhouette_score: {metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=sample_size):0.4f}")

init_setting = "random"
clf = KMeans(n_clusters=k, init=init_setting , n_init=10)
bench_k_means(estimator=clf, name=init_setting , data=data)

