import cv2
import cv2 as cv
import numpy as np
import warnings
from matplotlib import pyplot as plt
import pandas as pd
from numpy import double
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

#RESCALE
img = cv.imread(r'D:\Facultate\An 4\Licenta\Tema licenta\COVID-CT-master\Images-processed\COVID\2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89%0.png')

down_width = 300
down_height = 200
down_points = (down_width, down_height)
final_img = cv2.resize(img, down_points, interpolation = cv.INTER_LINEAR)


#NORMALIZATION

normalization_image = np.zeros((800, 800))

final_img = cv.normalize(final_img, normalization_image, 0, 255, cv.NORM_MINMAX)


#NOISE REMOVAL

final_img = cv.fastNlMeansDenoisingColored(final_img, None, 10, 10, 7, 21)

cv2.imwrite(r'D:\Facultate\An 4\Licenta\Tema licenta\COVID-CT-master\Images-processed\COVID\SALUT.png', final_img)

#3 LEAST SEMNIFICANT BITS TO 0
width = final_img.shape[1]
height = final_img.shape[0]

for x in range(height):
    for y in range(width):
         final_img[x,y] = final_img[x,y] & 248

grayImage = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)


#MEAN SHIFT ALG

index = pd.MultiIndex.from_product(
    (*map(range, grayImage.shape[:2]), ['l']),
    names=('row', 'col', None))
df_1 = pd.Series(grayImage.flatten(), index=index)
df_1 = df_1.unstack()
df_1 = df_1.reset_index().reindex(columns=['col','row','l'])

nd_1 = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_1)

ms = MeanShift(bandwidth = 0.08 , n_jobs=-1, bin_seeding=True, cluster_all=True).fit(nd_1)

labeled = ms.labels_

flat_image = grayImage.reshape((-1,1))

segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# get the average color of each segment
total = np.zeros((segments.shape[0], 1), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((grayImage.shape))

cv.imwrite(r'D:\Facultate\An 4\Licenta\Tema licenta\COVID-CT-master\Images-processed\COVID\SALUT1.png', result)

# SUPERPIXEL SEED

# show the result
# cv.imshow('result',result)
# cv.waitKey(0)
# cv.destroyAllWindows()

# seeds = cv.ximgproc.createSuperpixelSEEDS(width,height,1, 100, 4, 2, 5)
# color_img = np.zeros((height,width,1),np.uint8)
# color_img[:] = (0)
# seeds.iterate(result, 6)
#
# # retrieve the segmentation result
# labels = seeds.getLabels()
#
# # labels output: use the last x bits to determine the color
# num_label_bits = 2
# labels &= (1<<num_label_bits)-1
# labels *= 1<<(16-num_label_bits)
#
# mask = seeds.getLabelContourMask(False)
#
# # stitch foreground & background together
# mask_inv = cv.bitwise_not(mask)
# result_bg = cv.bitwise_and(result, result, mask=mask_inv)
# result_fg = cv.bitwise_and(color_img, color_img, mask=mask)
# result_img = cv.add(result_bg, result_fg)

# SUPERPIXEL SLIC

slic = cv2.ximgproc.createSuperpixelSLIC(result, region_size=20, ruler=100.0)
slic.iterate(24)
mask_slic = slic.getLabelContourMask()
label_slic = slic.getLabels()
number_slic = slic.getNumberOfSuperpixels()
mask_inv_slic = cv.bitwise_not(mask_slic)
imagine = cv.bitwise_and(result, result, mask=mask_inv_slic)

color_img = np.zeros((result.shape[0], result.shape[1], 1), np.uint8)
color_img[:] = (0)
result_ = cv2.bitwise_and(color_img, color_img, mask=mask_slic)
result_img = cv2.add(imagine, result_)

cv.imwrite(r'D:\Facultate\An 4\Licenta\Tema licenta\COVID-CT-master\Images-processed\COVID\SALUT2.png', result_img)