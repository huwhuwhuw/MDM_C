#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:55:35 2025

@author: huwtebbutt
"""

import numpy as np
import matplotlib.pyplot as plt
#from skimage import feature, transform, draw

from sklearn.cluster import DBSCAN

def DBScan_Cluster(points, eps=5, min_samples=10, Title='DBSCAN Clustering'):
    fig,ax=plt.subplots(dpi=500)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = points[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    ax.set_title(Title)
    ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    points = np.random.rand(1000, 2) * 100  # Scale points for better visualization
    DBScan_Cluster(points)



