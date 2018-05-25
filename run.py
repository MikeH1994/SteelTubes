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
        angles = angles[angles>3]#get rid of any element less than 3
        averageAngle = np.mean(angles)
        stdev = np.std(angles)
        return averageAngle,stdev
    except:
        return 0,0

def getAverageDisplacement(imageList,drawFlag = True):
    xDisplacement = []
    yDisplacement = []
    for i in range(len(imageList)-1):
        draw = drawFlag
        imgPath = imageList[i]
        imgPath2 = imageList[i+1]
        img1 = imgRead(imgPath)
        img2 = imgRead(imgPath2)
        imgMax1 = getNonzeroMax(img1)
        imgMax2 = getNonzeroMax(img2)
        height,width = img1.shape
        if imgMax1<5E4 or imgMax2<5E4:
            continue

        ind1 = np.where(img1>1)
        minX1,maxX1 = np.min(ind1[1]),np.max(ind1[1])
        minY1,maxY1 = np.min(ind1[0]),np.max(ind1[0])
        ind2 = np.where(img2>1)
        minX2,maxX2 = np.min(ind2[1]),np.max(ind2[1])
        minY2,maxY2 = np.min(ind2[0]),np.max(ind2[0])

        if (minY1>0 and minY2>0) or (maxY1<height-1 and maxY2<height-1):
            #if it is the beginning or the end of the tube, use this to get displacement
            pass
        else:
            continue
            #else go to next image
        dx,dy = 0,0

        if minY1>0 and minY2>0:
            dy = minY1-minY2
        elif maxY1<height-1 and maxY2<height-1:
            dy = maxY1-maxY2
        if minX1>0 and minX2>0:
            dx = minX1-minX2
        if dx>0 or dy<0:
            print dx,dy
            draw = True
        
        if not dy<0:
            yDisplacement.append(dy)
        if not dx>0:
            xDisplacement.append(dx)
        if draw:
            lineThickness = 2
            ax1 = plt.subplot(2,2,1)
            cv2.line(img1, (0, minY1), (width, minY1), 5E5, lineThickness)
            cv2.line(img1, (0, maxY1), (width, maxY1), 5E5, lineThickness)
            cv2.line(img1, (minX1, 0), (minX1, height), 5E5, lineThickness)
            #cv2.line(img1, (maxX1, 0), (maxX1, height), 5E5, lineThickness)
            ax1.imshow(img1)
            ax2 = plt.subplot(2,2,2)
            cv2.line(img2, (0, minY2), (width, minY2), 5E5, lineThickness)
            cv2.line(img2, (0, maxY2), (width, maxY2), 5E5, lineThickness)
            cv2.line(img2, (minX2, 0), (minX2, height), 5E5, lineThickness)
            #cv2.line(img2, (maxX2, 0), (maxX2, height), 5E5, lineThickness)
            ax2.imshow(img2)
            plt.show()
    if yDisplacement.size>0:
        yDisplacement = np.asarray(yDisplacement)
        deltaY = np.mean(yDisplacement)
        deltaYStd = np.std(yDisplacement)
        return deltaX,deltaXStd,deltaY,deltaYStd
      
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

def getDisplacement(dataRoot):
    rootDir = os.path.dirname(os.path.realpath(__file__))
    outpath = os.path.join(rootDir,"displacement.txt")
    f = open(outpath,'w')
    f.close()
    for folder in getFolderList(dataRoot)[0:15]:
        imageList = getFilesInFolder(folder)
        dx,dxStd,dy,dyStd = getAverageDisplacement(imageList)
        f = open(outpath,'a')
        print "{}\t{}\t{}\t{}\t{}".format(folder,dx,dxStd,dy,dyStd)
        f.write("{}\t{}\t{}\t{}\t{}\n".format(folder,dx,dxStd,dy,dyStd))
        f.close() 
        
        
if __name__ == "__main__":
    dir = "D:\Thermal"#"/home/mike/Documents/Code/Python/Thermal/"
    getDisplacement(dir)
