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

def getAverageDisplacement(imageList,draw = False):
    for i in range(len(imageList)-1):
        imgPath = imageList[i]
        imgPath2 = imageList[i+1]
        img1 = imgRead(imgPath)
        img2 = imgRead(imgPath2)
        imgMax1 = getNonzeroMax(img1)
        imgMax2 = getNonzeroMax(img2)

        if imgMax1<5E4 or imgMax2<5E4:
            continue
        ind = np.where(img>1) #get index of nonzero points
        minX,maxX = np.min(ind[1]),np.max(ind[1])
        minY,maxY = np.min(ind[0]),np.max(ind[0])

        if minY>0 or maxY<height-1:
            pass
        else:
            continue
        if draw:
            lineThickness = 1
            cv2.line(img, (0, minY), (width, minY), 5E5, lineThickness)
            cv2.line(img, (0, maxY), (width, maxY), 5E5, lineThickness)
            cv2.line(img, (minX, 0), (minX, height), 5E5, lineThickness)
            cv2.line(img, (maxX, 0), (maxX, height), 5E5, lineThickness)
            plt.imshow(img)
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

def getDisplacement(dataRoot):
    rootDir = os.path.dirname(os.path.realpath(__file__))
    outpath = os.path.join(rootDir,"displacement.txt")
    #f = open(outpath,'w')
    #f.close()
    for folder in getFolderList(dataRoot)[2:15]:
        imageList = getFilesInFolder(folder)
        getAverageDisplacement(imageList)
        #f = open(outpath,'a')
        #print "{}\t{}\t{}".format(folder,angle,stdev)
        #f.write("{}\t{}\t{}\n".format(folder,angle,stdev))
        #f.close() 
        
        
if __name__ == "__main__":
    dir = "D:\Thermal"#"/home/mike/Documents/Code/Python/Thermal/"
    getDisplacement(dir)
