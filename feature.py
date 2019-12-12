from skimage import feature
import numpy as np
import cv2
import math

class GetFeature:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius
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

        (x, y), ( ma,MA), angle = cv2.fitEllipse(cnt)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        compactness= (4*np.pi*area)/perimeter
        solidity = float(area) / hull_area
        eccen= math.sqrt(1 - (ma/MA)**2)
        compactness/= (compactness+solidity+eccen +eps)
        solidity/=(compactness+solidity+eccen +eps)
        eccen/=(compactness+solidity+eccen +eps)

        out = np.append(hist,[compactness,solidity,eccen] )
        return out





