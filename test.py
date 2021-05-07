import cv2
import numpy as np
import os
from libsvm.svmutil import *
from joblib import dump, load
from scipy.cluster.vq import vq

# Load model
classes_names, stdSlr, k, voc = load("./bin/bovw.pkl")
libsvm_model = svm_load_model('./bin/gender1.model')

# setup test data
test_path = './data/Validation'
test_name = os.listdir(test_path)
impath = []
imclass = []
class_id = 0

def imglist(dir_path):
  return [os.path.join(dir_path, name_file) for name_file in os.listdir(dir_path)][:300]

# Lấy danh sách hình và dánh nhãn cho chúng
for name in test_name:
  dir_path = os.path.join(test_path, name)
  class_path = imglist(dir_path=dir_path)
  impath+=class_path
  imclass+=[class_id]*len(class_path)
  class_id+=1

# đọc và rúc trích các đặt trưng của ảnh bằng BRISK
desc_list = []

brisk = cv2.BRISK_create()

for imname in impath:
  img = cv2.imread(imname)
  kpts, des = brisk.detectAndCompute(img, None)
  desc_list.append((imname, des))

# Loại bỏ các bức ảnh Lỗi
error_idx = []
for i in range(len(impath)):
  try:
    l = len(desc_list[i][1])
  except:
    error_idx.append(i)

error_idx.reverse()

for i in error_idx:
  impath.remove(impath[i])
  imclass.remove(imclass[i])
  desc_list.remove(desc_list[i])

test_features = np.zeros((len(impath), k), "float64")

for i in range(len(impath)):
  words, distance = vq(desc_list[i][1],voc)
  for w in words:
      test_features[i][w] += 1

nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(impath)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Đưa giá trị về [0..1]
test_features = stdSlr.transform(test_features)

p_label, p_acc, p_val = svm_predict(np.array(imclass), test_features, libsvm_model, '-b 1')

print('p_label: ', p_label)
print('p_acc: ', p_acc)
# print('p_val: ', p_val)