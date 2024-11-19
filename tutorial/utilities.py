import cv2
import numpy as np
from skimage import io
from pystackreg import StackReg
import pandas as pd
import scanpy as sc
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import multiprocessing as mp
import torch

def overlay_images(imgs, equalize=False, aggregator=np.mean):
    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]
    imgs = np.stack(imgs, axis=0)
    return aggregator(imgs, axis=0)

def composite_images(imgs, equalize=False, aggregator=np.mean):
    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = [img / img.max() for img in imgs]
    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))
    imgs = np.dstack(imgs)
    return imgs

def show_transformation(tmat, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    p = np.array([[1,120,1], [1,1,1], [250, 1, 1], [250,120,1], [1,120,1]])
    ax.plot(p[:, 0], p[:,1])
    q=np.dot(p, tmat.T)
    ax.plot(q[:, 0], q[:,1])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(['Original image', 'transformed image'])


def apply_transformation(row, sr):
    # Extract X and Y coordinates from the row
    x_ref, y_ref = row["X"], row["Y"]

    # Create a homogenous coordinate
    coordinate_ref = np.array([x_ref, y_ref, 1])

    # Apply the transformation matrix obtained from StackReg
    transformed_coordinate_mov = np.dot(sr.get_matrix(), coordinate_ref)

    # Extract the transformed X and Y coordinates
    x_mov, y_mov, _ = transformed_coordinate_mov

    # Update the row with the transformed coordinates
    row["X"], row["Y"] = x_mov, y_mov

    return row