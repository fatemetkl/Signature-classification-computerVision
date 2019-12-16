from feature import GetFeature
import os
import cv2

here = os.getcwd()

path = os.path.join(here, "images\\dataset", str(1))
cur_path=os.path.join(path,"11.tif")
gray = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2GRAY)
feature = GetFeature(20, 10)
print(feature.describe(gray))
path = os.path.join(here, "images\\dataset", str(0))
cur_path=os.path.join(path,"12.tif")
gray = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2GRAY)
feature = GetFeature(20, 10)
print(feature.describe(gray))
path = os.path.join(here, "images\\dataset", str(2))
cur_path=os.path.join(path,"13.tif")
gray = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2GRAY)
feature = GetFeature(20, 10)
print(feature.describe(gray))
path = os.path.join(here, "images\\dataset", str(3))
cur_path=os.path.join(path,"14.tif")
gray = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2GRAY)
feature = GetFeature(20, 10)
print(feature.describe(gray))