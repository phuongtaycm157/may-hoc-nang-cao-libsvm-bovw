import cv2
import numpy as np
import os
from libsvm.svmutil import *
from libsvm.svm import *
from joblib import dump, load
from scipy.cluster.vq import vq

# Load model
classes_names, stdSlr, k, voc = load("./bin/bovw.pkl")
libsvm_model = svm_load_model('./bin/gender1.model')

brisk = cv2.BRISK_create()

# Build Function

def read_img(img):
    kpts, des = brisk.detectAndCompute(img, None)
    vector = np.zeros(k, "float64")
    words, distance = vq(des,voc)
    for w in words:
      vector[w] += 1
    vector2d = stdSlr.transform([vector])
    return vector2d[0]

def classification_svm(img):
    vector = read_img(img)
    x0, max_idx = gen_svm_nodearray(vector)
    label = libsvm.svm_predict(libsvm_model, x0)
    return classes_names[int(label)]

# test

# img = cv2.imread('./data/Validation/female/112950.jpg.jpg')

# print(classification_svm(img))