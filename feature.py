from skimage import feature
import numpy as np
import cv2
import math
from skimage.measure import regionprops
class GetFeature:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def SkewKurtosis(self,img):
        h, w = img.shape
        x = range(w)  # cols value
        y = range(h)  # rows value
        # calculate projections along the x and y axes
        xp = np.sum(img, axis=0)
        yp = np.sum(img, axis=1)
        # centroid
        cx = np.sum(x * xp) / np.sum(xp)
        cy = np.sum(y * yp) / np.sum(yp)
        # standard deviation
        x2 = (x - cx) ** 2
        y2 = (y - cy) ** 2
        sx = np.sqrt(np.sum(x2 * xp) / np.sum(img))
        sy = np.sqrt(np.sum(y2 * yp) / np.sum(img))

        # skewness
        x3 = (x - cx) ** 3
        y3 = (y - cy) ** 3
        skewx = np.sum(xp * x3) / (np.sum(img) * sx ** 3)
        skewy = np.sum(yp * y3) / (np.sum(img) * sy ** 3)

        # Kurtosis
        x4 = (x - cx) ** 4
        y4 = (y - cy) ** 4
        # 3 is subtracted to calculate relative to the normal distribution
        kurtx = np.sum(xp * x4) / (np.sum(img) * sx ** 4) - 3
        kurty = np.sum(yp * y4) / (np.sum(img) * sy ** 4) - 3

        return skewx, skewy, kurtx, kurty

    def Ratio(self,img):
        a = 0
        for row in range(len(img)):
            for col in range(len(img[0])):
                if img[row][col]<1:
                    a = a + 1
        total = img.shape[0] * img.shape[1]
        return a / total
    def describe(self, image, eps=1e-7):

        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        ret, thresh = cv2.threshold(image, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]

        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        (x, y), ( MA,ma), angle = cv2.fitEllipse(cnt)
        #
        # hull = cv2.convexHull(cnt)
        # hull_area = cv2.contourArea(hull)
        #
        compactness= (4*np.pi*area)/pow(perimeter,2)
        # solidity = float(area) / hull_area
        # # eccen= math.sqrt(1 - (ma/MA)**2)
        #
        #
        # a = ma/2
        # b = MA/2
        #
        # eccentricity = math.sqrt(pow(a, 2)-pow(b, 2))
        # eccen = round(eccentricity/a, 2)
        #
        # compactness/= (compactness+solidity+eccen +eps)
        # solidity/=(compactness+solidity+eccen +eps)
        # eccen/=(compactness+solidity+eccen +eps)

        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area


        r = regionprops(image.astype("int8"))
        ratio = self.Ratio(image)
        skewx, skewy, kurtx, kurty =self.SkewKurtosis(image)
        out = np.append(hist,[compactness,r[0].eccentricity, r[0].solidity,skewx, skewy, kurtx, kurty,ratio,extent,angle,MA/ma] )
        return out





