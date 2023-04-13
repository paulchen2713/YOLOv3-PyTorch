# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import random
from tqdm import tqdm 
import sklearn.cluster as cluster
from sklearn import metrics
import pandas as pd

import time
from datetime import date
from functools import cmp_to_key


def iou(x, centroids):
    dists = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            dist = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            dist = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            dist = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w, h are bigger than c_w and c_h respectively
            dist = (c_w * c_h) / (w * h)
        dists.append(dist)
    return np.array(dists)


def avg_iou(x, centroids):
    n, d = x.shape
    sums = 0.0
    for i in range(x.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] slightly ineffective, but I am too lazy
        sums += max(iou(x[i], centroids))
    return sums / n


def write_anchors_to_file(centroids, distance, inertia, anchor_file):
    print(f"")
    print(f"Number of clusters: {len(centroids)}")
    print(f"Average IoU: {distance}")
    print(f"Inertia: {inertia}")
    print(f"Anchors: ")
    for i, centroid in enumerate(centroids):
        w, h = centroid[0], centroid[1]
        # print(f"{i + 1}: ({w}, {h})")
        print(f"{{{w:0.3f}, {h:0.3f}}}", end=', ')
        # print(f"({w}, {h})", end=', ')
    print(f"\n")
    
    with open(anchor_file, 'w') as f:
        print(f"Number of clusters: {len(centroids)}", file=f)
        print(f"Average IoU: {distance}", file=f)
        print(f"Inertia: {inertia}\n", file=f)

        print(f"Anchors original: ", file=f)
        for i, centroid in enumerate(centroids):
            w, h = centroid[0], centroid[1]
            print(f"({w}, {h})", end=', ', file=f)
            if (i + 1) % 3 == 0:
                print(f"", file=f)
        print(f"", file=f)

        print(f"Anchors rounded to 2 decimal places: ", file=f)
        for i, centroid in enumerate(centroids):
            w, h = centroid[0], centroid[1]
            print(f"({w:0.2f}, {h:0.2f})", end=', ', file=f)
        print(f"\n", file=f)
    
        print(f"Anchors rounded to 3 decimal places: ", file=f)
        for i, centroid in enumerate(centroids):
            w, h = centroid[0], centroid[1]
            print(f"({w:0.3f}, {h:0.3f})", end=', ', file=f)
        print(f"", file=f)
        
    print(f"Writing anchors to {anchor_file}.txt")



def k_means(x, n_clusters, eps):
    init_index = [random.randrange(x.shape[0]) for _ in range(n_clusters)]
    centroids = x[init_index]

    dist = old_dist = []
    iterations = 0
    diff = 1e10
    c, dim = centroids.shape

    while True:
        iterations += 1
        dist = np.array([1 - iou(i, centroids) for i in x])
        if len(old_dist) > 0:
            diff = np.sum(np.abs(dist - old_dist))

        print(f'diff = {diff}') # diff is a float

        if diff < eps or iterations > 1000:
            print(f"Number of iterations took = {iterations}") # 
            print("Centroids = ", centroids)
            return centroids

        # assign samples to centroids
        belonging_centroids = np.argmin(dist, axis=1)

        # calculate the new centroids
        centroid_sums = np.zeros((c, dim), np.float)
        for i in range(belonging_centroids.shape[0]):
            centroid_sums[belonging_centroids[i]] += x[i]

        for j in range(c):
            centroids[j] = centroid_sums[j] / np.sum(belonging_centroids == j)

        old_dist = dist.copy()


def get_file_content(fnm):
    with open(fnm) as f:
        return [line.strip() for line in f]


def sample(args):
    print("Reading Data ...")

    file_list = []
    for f in args.file_list:
        file_list.extend(get_file_content(f))

    data = []
    for one_file in tqdm(file_list):
        one_file = one_file.replace('images', 'labels') \
            .replace('JPEGImages', 'labels') \
            .replace('.png', '.txt') \
            .replace('.jpg', '.txt')
        for line in get_file_content(one_file):
            clazz, xx, yy, w, h = line.split()
            data.append([float(w),float(h)]) 

    data = np.array(data)
    if args.engine.startswith("sklearn"):
        if args.engine == "sklearn":
            km = cluster.KMeans(n_clusters=args.num_clusters, tol=args.tol, verbose=True)
        elif args.engine == "sklearn-mini":
            km = cluster.MiniBatchKMeans(n_clusters=args.num_clusters, tol=args.tol, verbose=True)
        km.fit(data)
        result = km.cluster_centers_
        # distance = km.inertia_ / data.shape[0]
        distance = avg_iou(data, result)
    else:
        result = k_means(data, args.num_clusters, args.tol)
        distance = avg_iou(data, result)

    write_anchors_to_file(result, distance, args.output)


def cmp_by_area(a, b):
    area_a = a[0] * a[1]
    area_b = b[0] * b[1]
    if area_a == area_b:
        return 0
    elif area_a < area_b:
        return -1 
    else: 
        return 1


def bench_KMeans(estimator, data, anchor_file, show=False):
    tic = time.perf_counter()
    estimator.fit(data) # the clustering result may be differnt each time
    toc = time.perf_counter()
    duration = toc - tic
    # print(estimator)    # e.g. KMeans(n_clusters=9, verbose=True)

    result = estimator.cluster_centers_
    # print(type(result))  # <class 'numpy.ndarray'>
    # print(result.shape)  # (9, 2)
    # print(f"{len(data)}, {len(estimator.labels_)}, {len(result)}")  # 7086, 7086, 9

    inertia = estimator.inertia_
    silhouette_score = metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=len(data))
    distance = avg_iou(data, result)
    
    if show == True:
        print(100 * '-')
        print(f"Estimator Settings: {estimator}")
        print(f"Number of Clusters: {len(result)}")
        print(f"Average IoU: {distance}")
        print(f"Inertia: {inertia}")
        print(f"Silhouette Score: {silhouette_score}")
        print(f"Date and Duration: {date.today()} / {duration:0.4f} seconds")
        print(f"Anchors: ")

    centroids = result.tolist()
    # print(len(centroids))      # 9
    # print(len(centroids[0]))   # 2
    # print(type(centroids))     # <class 'list'>
    # print(type(centroids[0]))  # <class 'list'>

    cmp_key = cmp_to_key(cmp_by_area)
    centroids.sort(key=cmp_key)

    if show == True: 
        for i, centroid in enumerate(centroids):
            w, h = centroid
            # Print out the width and height (w, h) of an anchor along with its cross-sectional area multiplied by 10000. 
            # This will help us distinguish between high and low anchor values easily.
            print(f"  {i + 1}: ({w:0.20f}, {h:0.20f})   {w*h*10000:<12}")

    with open(anchor_file, 'w') as f:
        print(f"Estimator: {estimator}", file=f)
        print(f"Number of Clusters: {len(result)}", file=f)
        print(f"Average IoU: {distance}", file=f)
        print(f"Inertia: {inertia}", file=f)
        print(f"Silhouette Score: {silhouette_score}", file=f)
        print(f"Date and Duration: {date.today()} / {duration:0.4f} seconds\n", file=f)

        print(f"Anchors: ", file=f)
        for i, centroid in enumerate(centroids):
            w, h = centroid
            # Print out the width and height (w, h) of an anchor along with its cross-sectional area multiplied by 10000. 
            # This will help us distinguish between high and low anchor values easily.
            print(f"  {i + 1}: ({w:0.20f}, {h:0.20f})   {w*h*10000:<12}", file=f)
        print(f"", file=f)

        print(f"Anchors original: ", file=f)
        for i, centroid in enumerate(centroids):
            w, h = centroid
            print(f"({w}, {h})", end=', ', file=f)
            if (i + 1) % 3 == 0:
                print(f"", file=f)
        print(f"", file=f)

        print(f"Anchors rounded to 2 decimal places: ", file=f)
        for i, centroid in enumerate(centroids):
            w, h = centroid
            print(f"({w:0.2f}, {h:0.2f})", end=', ', file=f)
            if (i + 1) % 3 == 0:
                print(f"", file=f)
        print(f"", file=f)
    
        print(f"Anchors rounded to 3 decimal places: ", file=f)
        for i, centroid in enumerate(centroids):
            w, h = centroid
            print(f"({w:0.3f}, {h:0.3f})", end=', ', file=f)
            if (i + 1) % 3 == 0:
                print(f"", file=f)
        print(f"", file=f)
        
    print(f"Writing anchors to {anchor_file}.txt")
    if show == True: print(100 * '-')


if __name__ == "__main__":
    tic = time.perf_counter()

    file_name = f"D:/Datasets/RADA/RD_JPG/width_heights.txt"
    DATASET = f"D:/Datasets/RADA/RD_JPG/"

    annotations = pd.read_csv(DATASET + f"train.csv").iloc[:, 1]  # fetch all the *.txt label names
    # for annotation in annotations: print(annotation)  # 1.txt ... 6000.txt

    data = []
    for annotation in annotations:
        with open(DATASET + "labels/" + annotation) as f:
            for line in f.readlines():
                class_index, xx, yy, w, h = line.split()
                data.append([float(w), float(h)])
    data = np.array(data)
    # print(data.shape) # (7086, 2)
    
    num_clusters = 9
    tol = 0.0001  # 0.005

    sklearnKMeans = cluster.KMeans(n_clusters=num_clusters, tol=tol, verbose=True)
    file_name1 = DATASET + f"anchors-sklearn-KMeans.txt"
    # bench_KMeans(estimator=sklearnKMeans, data=data, anchor_file=file_name1, show=False)
    
    miniBatchKMeans = cluster.MiniBatchKMeans(n_clusters=num_clusters, tol=tol, verbose=True)
    file_name2 = DATASET + f"anchors-miniBatch-KMeans.txt"
    # bench_KMeans(estimator=miniBatchKMeans, data=data, anchor_file=file_name2, show=False)
    
    
    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")


