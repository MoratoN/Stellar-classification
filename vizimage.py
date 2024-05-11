# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 09:00:10 2024

@author: naomi morato
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
# %matplotlib inline
from astropy.visualization import make_lupton_rgb
path='C:/Users/naomi/OneDrive/Documents/Python Scripts/visual transformers based stellar classification/stellar-classification-main/stellar_classification_dataset/test/'
#img_array = np.load(path+'O/1333.npy')
#img_array = np.load(path+'B/170.npy')
#img_array = np.load(path+'A/36.npy')
#img_array = np.load(path+'F/5313.npy')
#img_array = np.load(path+'G/54.npy')
#img_array = np.load(path+'K/39.npy')
img_array = np.load(path+'M/161.npy')


w = 64
h = 64
x = np.zeros((w,h, 6))

# u (λ = 0.355 μm), g (λ = 0.477 μm), r (λ = 0.623 μm), i (λ = 0.762 μm), and z (λ = 0.762 μm)

gri = make_lupton_rgb(img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3], Q=8, stretch=0.5)
urz = make_lupton_rgb(img_array[:, :, 0], img_array[:, :, 2], img_array[:, :, 4], Q=8, stretch=0.5)

gri_1 = cv2.cvtColor(gri, cv2.COLOR_BGR2RGB)
gri_1 = Image.fromarray(gri_1)

urz_1 = cv2.cvtColor(urz, cv2.COLOR_BGR2RGB)
urz_1 = Image.fromarray(urz)

enh_bri = ImageEnhance.Brightness(gri_1)
gri_brightened = enh_bri.enhance(0.8)
gri_brightened = cv2.cvtColor(np.array(gri_brightened), cv2.COLOR_RGB2BGR)

enh_bri = ImageEnhance.Brightness(urz_1)
urz_brightened = enh_bri.enhance(0.8)
urz_brightened = cv2.cvtColor(np.array(urz_brightened), cv2.COLOR_RGB2BGR)

gri = cv2.resize(gri, (w, h))
urz = cv2.resize(urz, (w, h))


cv2.imshow("GRI-Channels", gri)
cv2.imshow("URZ-Channels", urz)


x[:, :, 0:3] = gri
x[:, :, 3:6] = urz

x = np.transpose(x / 255, (2, 0, 1)).astype(np.float32)

#plt.imshow(gri, cmap='gray')
#plt.show()
plt.imshow(urz, cmap='gray')
plt.show()



