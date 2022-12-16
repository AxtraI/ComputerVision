import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_triangle
 
def circularity(region, label=1):
    return (region.perimeter**2) / region.area

def toGray(image):
    return (0.2989 * image[:, :, 0] + 0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]).astype("uint8")

def binarisation(image, limit_min, limit_max):
    B = image.copy()
    B[B < limit_min] = 0
    B[B >= limit_max] = 0
    B[B > 0] = 1
    return B
 
img = 1
pencils = 0 
fig = plt.figure()
 
for img in range(1, 13):

    image = plt.imread("C:\\Users\\fsien\\OneDrive\\Desktop\\Pencils\\images\\img ("+str(img)+").jpg")

    gray = toGray(image)
    thresh = threshold_triangle(gray)
    binary = binarisation(gray, 0, thresh)
    binary = morphology.binary_dilation(binary, iterations=1)
    labeled = label(binary)
 
    areas = []
 
    for region in regionprops(labeled):
        areas.append(region.area)
 
    for region in regionprops(labeled):
        if region.area < np.mean(areas):
            labeled[labeled == region.label] = 0
        bbox = region.bbox
        if bbox[0] == 0 or bbox[1] == 0:
            labeled[labeled == region.label] = 0
 
    labeled[labeled > 0] = 1
    labeled = label(labeled)
 
    for i, region in enumerate(regionprops(labeled)):
        if (((circularity(region, i) > 105) and (500000 > region.area > 300000))):
            pencils += 1

    fig.add_subplot(4, 3, img)
    plt.imshow(labeled)
 
print("Количество карандашей:", pencils)
plt.show() 
