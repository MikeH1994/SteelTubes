import cv2
import numpy as np
import os
import tkFileDialog
import warnings
from Tkinter import *
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import quad

def getTransformationECC(img1,img2):
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 5000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (img1,img2,warp_matrix, warp_mode, criteria)
    return warp_matrix

def getBoundingRectangle(img):
    im2,contours,hierarchy = cv2.findContours(img, 1, 2)
    cnt = contours[0]
    x0,y0,w,h = 0,0,0,0

    maxW,maxH = 0,0
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x0,y0,w,h = rect
        if w>20 and h>20:
            return x0,y0,w,h
    print "BOUNDING BOX NOT FOUND"

    return x0,y0,w,h
    
def cosTheta(arr1,arr2):
    dotProduct = 0
    mag = np.linalg.norm(arr1)*np.linalg.norm(arr2)
    if mag==0:
        return 0
    for i in range(len(arr1)):
        dotProduct+=np.dot(arr1[i],arr2[i])
    #dotProduct/=mag
    return dotProduct
    
    
    
def getTransformationManual(img1,img2):

    bestCosTheta = 0
    imgHeight,imgWidth = img1.shape

    _x0_1,_y0_1,bbWidth1,bbHeight1 = getBoundingRectangle(img1)
    bBox1 = img1[_y0_1:_y0_1+bbHeight1,_x0_1:_x0_1+bbWidth1]
    _x0_2,_y0_2,bbWidth2,bbHeight2 = getBoundingRectangle(img2)
    bBox2 = img2[_y0_2:_y0_2+bbHeight2,_x0_2:_x0_2+bbWidth2]
    
    
    startX,startY = 0,0
    bestXPos,bestYPos = 0,0
    
    
    
    #move the contour region in img2 over img1
    #i,j represent top left corner coordinates
    for i in range(-20,20):
        for j in range(60):
            x0_1,x1_1,y0_1,y1_1 = 0,0,0,0
            x0_2,x1_2,y0_2,y1_2 = 0,0,0,0
            if i<0:
                x0_1 = -i
                x1_1 = min(bbWidth2+i,bbWidth1)
                x0_2 = 0
                x1_2 = min(bbWidth1+i,bbWidth2)
            else:
                x0_1 = 0
                x1_1 = min(bbWidth2-i,bbWidth1)
                x0_2 = i
                x1_2 = min(bbWidth1-i,bbWidth2)
            
            if j<0:
                y0_1 = -j
                y1_1 = min(j+bbHeight1,j+bbHeight2)                
                y0_2 = 0
                y1_2 = min(j+bbHeight1,bbHeight2)
            else:
                y0_1 = 0
                y1_1 = min(bbHeight2-i,bbHeight1)
                y0_2 = j
                y1_2 = min(bbHeight1-i,bbHeight2)
            if x0_1<0 or x0_2<0 or y0_1<0 or y0_2<0:
                print "ERROR SETTING BOUNDS"
                exit()
            print bbWidth1,bbHeight1,bbWidth2,bbHeight2
            print x0_1,x1_1,y0_1,y1_1
            print x0_2,x1_2,y0_2,y1_2
            arr1 = img1[y0_1:y1_1,x0_1:x1_1]
            arr2 = img2[y0_2:y1_2,x0_2:x1_2]
            
            plt.subplot(121)
            plt.imshow(arr1)
            plt.subplot(122)
            plt.imshow(arr2)
            plt.show()
            
            p = cosTheta(arr1,arr2)

            if p>bestCosTheta:
                bestCosTheta = p
                bestXPos = i+_x0_1
                bestYPos = j+_y0_1
    arrWidth = min(imgWidth-bestXPos,bbWidth1)
    arrHeight = min(imgHeight-bestYPos,bbHeight1)
    arr_1 = img1[_y0_1:_y0_1+bbHeight1,_x0_1:_x0_1+bbWidth1]
    arr_2 = img2[bestYPos:bestYPos+arrHeight,bestXPos:bestXPos+arrWidth]

    print bestXPos,bestYPos,bestCosTheta
    return bestXPos,bestYPos,bestCosTheta

def imgRead(imgPath):
    img = cv2.imread(imgPath,-1)
    thresh = getOTSUThreshold(img)
    thresh,img = cv2.threshold(img,thresh,0,cv2.THRESH_TOZERO)
    return img

def convertTo8Bit(img,imgMin=None,imgMax = None):
    img64 = img.astype(np.float64)
    
    if imgMin == None:
        imgMin = getNonzeroMin(img64)
    if imgMax == None:
        imgMax = getNonzeroMax(img64)

    img64 = (img64-imgMin)/(imgMax-imgMin)
    img64 = img64.clip(min=0)
    img64 = 255 * img64
    img8 = img64.astype(np.uint8)
    return img8
    
def convertImgsTo8Bit(img1,img2):
    imgMin = min(getNonzeroMin(img1),getNonzeroMin(img2))
    imgMax = max(getNonzeroMax(img1),getNonzeroMax(img2))
    img1 = convertTo8Bit(img1,imgMin = imgMin,imgMax = imgMax)
    img2 = convertTo8Bit(img2,imgMin = imgMin,imgMax = imgMax)
    return img1,img2
   
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]    
    
def getNonzeroMin(img):
    return np.min(img[np.nonzero(img)])

def getNonzeroMax(img):
    return np.max(img[np.nonzero(img)])
    
def getFolderList(a_dir):
    directories = []
    for path,subdirectories,files in os.walk(a_dir):
        for subdir in subdirectories:
            subpath = os.path.join(path, subdir)
            contents = os.listdir(subpath)
            imageList = [os.path.join(subpath, i) for i in contents if ".tiff" in i]
            if len(imageList)>0:
                directories.append(subpath)
    return directories

def getFilesInFolder(folderpath):
    contents = os.listdir(folderpath)
    files =  [os.path.join(folderpath, i) for i in contents if ".tiff" in i]
    return sorted(files, key=str.lower)
    
def convertImageToTemperature(img,par):
    img = fit(img,*par)
    return img
    
def getOTSUThreshold(img,min = 0,max=6E4,nBins = 512):
    #shamelessly c&p'd from https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    #find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([img],[0],None,[nBins],[min,max])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(nBins)
    fn_min = np.inf
    thresh = -1
    for i in xrange(1,nBins):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[-1]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    thresh = min + thresh*(max-min)/nBins
    return thresh

def fit(x,a,b,c,d):
    return a + b*x + c*x**2 + d*x**3
    
def gaussian(x,a,mu,c):
    return a*np.exp(-(x-mu)**2 / c)        

def gaussianFixedMu(mu):
    def fit(x,a,c):
        return gaussian(x,a,mu,c)
    return fit

def getDLtoTemperatureLUT():
    dir = os.path.dirname(os.path.realpath(__file__))
    path =  os.path.join(dir,"digitalToTemperature.txt")
    data = np.genfromtxt(path)
    DL = data[:,0]
    temperature = data[:,1]
    
    LUT = interp1d(DL,temperature,fill_value='extrapolate')
    return LUT

def getFittingPar():
    dir = os.path.dirname(os.path.realpath(__file__))
    path =  os.path.join(dir,"digitalToTemperature.txt")
    data = np.genfromtxt(path)
    DL = data[:,0]
    temperature = data[:,1]
    
    coeffs, matcov = curve_fit(fit, DL, temperature, (1,1,1,1))

    return coeffs        
