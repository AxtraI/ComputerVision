import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage import morphology

def count_objects(image, mask):
    erosion = morphology.binary_erosion(image, mask)
    dilation = morphology.binary_dilation(erosion, mask)
    image -= dilation
    count = label(dilation).max()
    return count

image = np.load('C:\\ps.npy.txt')

masks = np.array([
    np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])
], dtype=object)

all_objects = 0
i = 0

for mask in masks:
    count = count_objects(image, mask)
    all_objects += count
    i += 1
    print(f'Count objects {i}:', count)
print('Count of all objects:', all_objects)
