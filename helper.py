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

def imgRead(imgPath):
    img = cv2.imread(imgPath,-1)
    thresh = getOTSUThreshold(img)
    thresh,img = cv2.threshold(img,thresh,0,cv2.THRESH_TOZERO)
    return img

def convertTo8Bit(img):
    img64 = img.astype(np.float64)
    imgMax = getNonzeroMax(img64)
    imgMin = getNonzeroMin(img64)
    img64 = (img64-imgMin)/(imgMax-imgMin)
    img64 = img64.clip(min=0)
    img64 = 255 * img64
    img8 = img64.astype(np.uint8)
    return img8
    
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
  
def imread(path,process = False,par = None):
    img = cv2.imread(path,-1).astype(np.float32)
    if process:
        img = convertImageToTemperature(img,par)
        thresh = getOTSUThreshold(img,min = 400, max = 900)
        thresh,img = cv2.threshold(img,thresh,0,cv2.THRESH_TOZERO)
    return img
    
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
