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
from helper import *

averageRotation = 3.0802

def getAverageAngle(imageList,draw = False):
    try:
        angles = []
        for imgPath in imageList:
            img = imgRead(imgPath)
            img8 = convertTo8Bit(img)
            imgMax = getNonzeroMax(img)
            if imgMax<5E4:
                continue
            edges = cv2.Canny(img8,50,50)
            lines = cv2.HoughLines(edges,1,np.pi/180,20)
            if not lines is None:
                angles = angles+[theta for rho,theta in lines[0]]
            if draw:
                for rho,theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(img,(x1,y1),(x2,y2),5E6,2)
                plt.imshow(img)
                plt.show()        
        angles = np.asarray(angles)
        angles = angles[angles>3]
        averageAngle = np.mean(angles)
        stdev = np.std(angles)
        return averageAngle,stdev
    except:
        return 0,0

def getDisplacement(img8_1,img8_2):
    """warp_matrix = getTransformationECC(img8_1,img8_2)
    useECC = True
    dx,dy = 0,0
    dx,dy = warp_matrix[0][2],warp_matrix[1][2]
    dy*=-1
    dx*=-1
    dx = int (dx)
    dy = int (dy)
    """
    dx,dy,theta = getTransformationManual(img8_1,img8_2)
    return dx,dy

def mergeImage(imageList,drawFlag = False):
    xDisplacement = []
    yDisplacement = []
    deltaX,deltaXStd,deltaY,deltaYStd = 0,0,0,0
    mergedImageWidth = 4000
    mergedImageHeight = 4000
    mergedImage = np.zeros((mergedImageHeight,mergedImageWidth)).astype(np.float64)
    mergedImageCount = np.zeros((mergedImageHeight,mergedImageWidth)).astype(np.float64)
    x,y = 200,200
    #print mergedImage.shape,mergedImageCount.shape
    img0 = imgRead(imageList[0])
    h,w = img0.shape
    mergedImage[y:y+h,x:x+w]+=img0
    thresh, binaryImg0 = cv2.threshold(img0, 1, 1, cv2.THRESH_BINARY);
    mergedImageCount[y:y+h,x:x+w]+=binaryImg0
    for i in range(1,8):#mergedImageHeight,mergedImageWidth
        draw = drawFlag
        imgPath = imageList[i]
        imgPath2 = imageList[i+1]
        img1 = imgRead(imgPath)
        img2 = imgRead(imgPath2)
        imgMax1 = getNonzeroMax(img1)
        imgMax2 = getNonzeroMax(img2)
        if imgMax1<5E4 or imgMax2<5E4:
            continue

        img8_1,img8_2 = convertImgsTo8Bit(img1,img2)
        dx,dy = getDisplacement(img8_1,img8_2)
        x+=dx
        y+=dy
        #print img1.shape,img2.shape,mergedImage.shape,y,y+h,x,x+w
        mergedImage[y:y+h,x:x+w]+=img2
        thresh, binaryImg = cv2.threshold(img8_2, 1, 1, cv2.THRESH_BINARY);
        mergedImageCount[y:y+h,x:x+w]+=binaryImg
    mergedImage[np.where(mergedImageCount>0)]/=mergedImageCount[np.where(mergedImageCount>0)]
    plt.imshow(mergedImage)
    plt.show()
    
def getAngles(dataRoot):
    rootDir = os.path.dirname(os.path.realpath(__file__))
    outpath = os.path.join(rootDir,"angles.txt")
    f = open(outpath,'w')
    f.close()
    for folder in getFolderList(dataRoot):
        imageList = getFilesInFolder(folder)
        angle,stdev = getAverageAngle(imageList)
        f = open(outpath,'a')
        print "{}\t{}\t{}".format(folder,angle,stdev)
        f.write("{}\t{}\t{}\n".format(folder,angle,stdev))
        f.close()

def mergeImages(dataRoot):
    rootDir = os.path.dirname(os.path.realpath(__file__))
    outpath = os.path.join(rootDir,"displacement.txt")
    f = open(outpath,'w')
    f.close()
    for folder in getFolderList(dataRoot):
        imageList = getFilesInFolder(folder)
        mergeImage(imageList)

        
if __name__ == "__main__":
    dir = "/home/mike/Documents/Code/Python/Thermal/"#"D:\Thermal"#
    mergeImages(dir)
