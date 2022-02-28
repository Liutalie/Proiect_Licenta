import cv2
import cv2 as cv
import numpy as np
import warnings
from matplotlib import pyplot as plt
import pandas as pd
from numpy import double
from skimage import segmentation
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

slic = cv2.ximgproc.createSuperpixelSLIC(result,algorithm=100, region_size=25, ruler=100.0)
slic.iterate(24)
mask_slic = slic.getLabelContourMask() # Contine: val min/max a unui pixel (0,255), shape (200X300), size (60000), array (conturul separator)
label_slic = slic.getLabels() # Contine: min/max al zonei (0->144 zone), shape (200X300), size (60000), array (fiecare pixel a carei zone apartine)
number_slic = slic.getNumberOfSuperpixels() # Retine numarul de SuperPixeli
mask_inv_slic = cv.bitwise_not(mask_slic) # Inversam bitii
imagine = cv.bitwise_and(result, result, mask=mask_inv_slic) # SI logic, punem in img conturul SuperPixelilor

color_img = np.zeros((result.shape[0], result.shape[1], 1), np.uint8) # Contine: shpe (200,300,1 (cate culori)), size (60000), array (fiecare pixel)
color_img[:] = (0)
result_ = cv2.bitwise_and(color_img, color_img, mask=mask_slic)
result_img = cv2.add(imagine, result_)


# Mean of each SuperPixel (Trebuie sa ma mai uit putin)
im_rp = result_img.reshape((result_img.shape[0]*result_img.shape[1]))
sli_1d = np.reshape(label_slic, -1)
uni = np.unique(sli_1d)
uu = np.zeros(im_rp.shape)
for i in uni:
    loc = np.where(sli_1d == i)[0]
    mm = np.mean(im_rp[loc], axis=0)
    uu[loc] = mm
temporary_img = np.reshape(uu,[result_img.shape[0], result_img.shape[1]]).astype('uint8')
temporary_img = cv2.bitwise_and(temporary_img, mask_inv_slic)

cv.imwrite(r'D:\Facultate\An 4\Licenta\Tema licenta\COVID-CT-master\Images-processed\COVID\SALUT2.png', temporary_img)