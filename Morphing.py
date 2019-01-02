import numpy as np
from os import path
from scipy import spatial, interpolate, ndimage
import imageio
import os
import glob
import re
from PIL import Image, ImageDraw

class Affine:
    def __init__(self, source, destination):
        # Assuming source and destimations indices are in x,y
        if(source.dtype != np.float64 or source.shape != (3,2)):
            raise ValueError("Input Matrices should of size 3x2 and only contian float value")
        if(destination.dtype != np.float64 or destination.shape != (3,2)):
            raise ValueError("Input Matrices should of size 3x2 and only contian float value")
        self.source = source
        self.destination = destination
        self.matrix = self.createHMatrix()
    
    # Creating and solving equation to get the transformation matrix

    def createHMatrix(self):
        """
        Return Value: numpy array of the affine Transformation matrix
        """
        # ASSUMING INPUT IS ROWS, COLUMNS
        A = np.array([
                    [self.source[0][0],self.source[0][1],1,0,0,0],
                    [0,0,0,self.source[0][0],self.source[0][1],1],
                    [self.source[1][0],self.source[1][1],1,0,0,0],
                    [0,0,0,self.source[1][0],self.source[1][1],1],
                    [self.source[2][0],self.source[2][1],1,0,0,0],
                    [0,0,0,self.source[2][0],self.source[2][1],1]],float)
        b = np.array([
                    [self.destination[0][0]],
                    [self.destination[0][1]],
                    [self.destination[1][0]],
                    [self.destination[1][1]],
                    [self.destination[2][0]],
                    [self.destination[2][1]]],float)
        h = np.linalg.solve(A,b)
        H = np.array([
            [h[0],h[1],h[2]],
            [h[3],h[4],h[5]],
            [0,0,1]],float)
        return H
        
    def transform(self,sourceImage, destinationImage):
        if not isinstance(sourceImage,np.ndarray) or not isinstance(destinationImage,np.ndarray):
            raise ValueError("Given values need to a 2D numpy array")
        
        point1 = (self.destination[0][1],self.destination[0][0])
        point2 = (self.destination[1][1],self.destination[1][0])
        point3 = (self.destination[2][1],self.destination[2][0])
        # Calculate the indices
        
        triVertices = (point1,point2,point3)
        img = Image.new('L',(sourceImage.shape),0)
        ImageDraw.Draw(img).polygon(triVertices,outline=255,fill=255)
        mask = np.array(img)
        indicesinTriangle = np.transpose(np.nonzero(mask))
        row = indicesinTriangle[:,1]
        column = indicesinTriangle[:,0]
        AffineInv = np.linalg.inv(self.matrix)
        DestinationPoints = np.dot(AffineInv,np.vstack((column,row,np.ones(len(row)))))
        destinationImage[row, column] = ndimage.map_coordinates(sourceImage,[DestinationPoints[1], DestinationPoints[0]],order=1)
        
class Blender:
    def __init__(self,startImage,startPoints,endImage,endPoints):
        if (not isinstance(startImage,np.ndarray) or
            not isinstance(startPoints,np.ndarray) or
            not isinstance(endImage,np.ndarray) or
            not isinstance(endPoints,np.ndarray)):
            raise TypeError('Input should be a numpy array')
        self.startImage = startImage
        self.startPoints = startPoints
        self.endImage = endImage
        self.endPoints = endPoints
    
    def getBlendedImage(self,alpha):
        trianglePoints = spatial.Delaunay(self.endPoints).vertices
        sourceTarget = np.zeros(self.startImage.shape,dtype=np.uint8)
        endTarget = np.zeros(self.endImage.shape,np.uint8)
        for vertices in trianglePoints:
            # ALL COORDINATES ARE IN X,Y
            sourceP1 = (self.startPoints[vertices[0]][0],self.startPoints[vertices[0]][1])
            sourceP2 = (self.startPoints[vertices[1]][0],self.startPoints[vertices[1]][1])
            sourceP3 = (self.startPoints[vertices[2]][0],self.startPoints[vertices[2]][1])
            endP1 = (self.endPoints[vertices[0]][0],self.endPoints[vertices[0]][1])
            endP2 = (self.endPoints[vertices[1]][0],self.endPoints[vertices[1]][1])
            endP3 = (self.endPoints[vertices[2]][0],self.endPoints[vertices[2]][1])
            destP1 = ((1-alpha) * sourceP1[0] + alpha * endP1[0],(1-alpha) * sourceP1[1] + alpha * endP1[1])
            destP2 = ((1-alpha) * sourceP2[0] + alpha * endP2[0],(1-alpha) * sourceP2[1] + alpha * endP2[1])
            destP3 = ((1-alpha) * sourceP3[0] + alpha * endP3[0],(1-alpha) * sourceP3[1] + alpha * endP3[1])
            sourceAffine = Affine(np.array([sourceP1,sourceP2,sourceP3]),np.array([destP1,destP2,destP3]))
            sourceAffine.transform(self.startImage,sourceTarget)
            endAffine = Affine(np.array([endP1,endP2,endP3]),np.array([destP1,destP2,destP3]))
            endAffine.transform(self.endImage,endTarget)
        return np.array((1-alpha) * sourceTarget +  alpha * endTarget,dtype=np.uint8)

    def generateMorphVideo(self,targetFolderPath, sequenceLength, includeReversed):
        if not os.path.isdir(targetFolderPath):
            os.makedirs(targetFolderPath)
        alphaValues = list(np.arange(0,1,1/(sequenceLength - 1)))
        alphaValues.append(1)
        if(includeReversed):
            alphaValues = list(alphaValues) + list(reversed(alphaValues)) 
        video = imageio.get_writer(targetFolderPath + '/morph.mp4',fps=5)
        frameCount = 1
        for alpha in alphaValues:
            blendImage = self.getBlendedImage(alpha)
            imageName = 'frame{:03d}.jpg'.format(frameCount)
            imagePath = path.join(targetFolderPath,imageName)
            imageio.imwrite(imagePath,blendImage)
            video.append_data(imageio.imread(imagePath))
            frameCount += 1
        video.close()

class ColorAffine:
    def __init__(self, source, destination):
        # Assuming source and destimations indices are in x,y
        if(source.dtype != np.float64 or source.shape != (3,2)):
            raise ValueError("Input Matrices should of size 3x2 and only contian float value")
        if(destination.dtype != np.float64 or destination.shape != (3,2)):
            raise ValueError("Input Matrices should of size 3x2 and only contian float value")
        self.source = source
        self.destination = destination
        self.matrix = self.createHMatrix()
    
    # Creating and solving equation to get the transformation matrix

    def createHMatrix(self):
        """
        Return Value: numpy array of the affine Transformation matrix
        """
        # ASSUMING INPUT IS ROWS, COLUMNS
        A = np.array([
                    [self.source[0][0],self.source[0][1],1,0,0,0],
                    [0,0,0,self.source[0][0],self.source[0][1],1],
                    [self.source[1][0],self.source[1][1],1,0,0,0],
                    [0,0,0,self.source[1][0],self.source[1][1],1],
                    [self.source[2][0],self.source[2][1],1,0,0,0],
                    [0,0,0,self.source[2][0],self.source[2][1],1]],float)
        b = np.array([
                    [self.destination[0][0]],
                    [self.destination[0][1]],
                    [self.destination[1][0]],
                    [self.destination[1][1]],
                    [self.destination[2][0]],
                    [self.destination[2][1]]],float)
        h = np.linalg.solve(A,b)
        H = np.array([
            [h[0],h[1],h[2]],
            [h[3],h[4],h[5]],
            [0,0,1]],float)
        return H

    def transform(self,sourceImage, destinationImage):
        if not isinstance(sourceImage,np.ndarray) or not isinstance(destinationImage,np.ndarray):
            raise ValueError("Given values need to a 2D numpy array")
        
        point1 = (self.destination[0][1],self.destination[0][0])
        point2 = (self.destination[1][1],self.destination[1][0])
        point3 = (self.destination[2][1],self.destination[2][0])
        # Calculate the indices
        
        triVertices = (point1,point2,point3)
        img = Image.new('L',(sourceImage.shape[0],sourceImage.shape[1]),0)
        ImageDraw.Draw(img).polygon(triVertices,outline=255,fill=255)
        mask = np.array(img)
        indicesinTriangle = np.transpose(np.nonzero(mask))
        row = indicesinTriangle[:,1]
        column = indicesinTriangle[:,0]
        AffineInv = np.linalg.inv(self.matrix)
        DestinationPoints = np.dot(AffineInv,np.vstack((column,row,np.ones(len(row)))))
        print(DestinationPoints.shape)
        destinationImage[row, column, 0] = ndimage.map_coordinates(sourceImage[:,:,0],[DestinationPoints[1], DestinationPoints[0]],order=1)
        destinationImage[row, column, 1] = ndimage.map_coordinates(sourceImage[:,:,1],[DestinationPoints[1], DestinationPoints[0]],order=1)
        destinationImage[row, column, 2] = ndimage.map_coordinates(sourceImage[:,:,2],[DestinationPoints[1], DestinationPoints[0]],order=1)
        
class ColorBlender:
    def __init__(self,startImage,startPoints,endImage,endPoints):
        if (not isinstance(startImage,np.ndarray) or
            not isinstance(startPoints,np.ndarray) or
            not isinstance(endImage,np.ndarray) or
            not isinstance(endPoints,np.ndarray)):
            raise TypeError('Input should be a numpy array')
        self.startImage = startImage
        self.startPoints = startPoints
        self.endImage = endImage
        self.endPoints = endPoints
    
    def getBlendedImage(self,alpha):
        trianglePoints = spatial.Delaunay(self.endPoints).vertices
        sourceTarget = np.zeros(self.startImage.shape,dtype=np.uint8)
        endTarget = np.zeros(self.endImage.shape,np.uint8)
        for vertices in trianglePoints:
            # ALL COORDINATES ARE IN X,Y
            sourceP1 = (self.startPoints[vertices[0]][0],self.startPoints[vertices[0]][1])
            sourceP2 = (self.startPoints[vertices[1]][0],self.startPoints[vertices[1]][1])
            sourceP3 = (self.startPoints[vertices[2]][0],self.startPoints[vertices[2]][1])
            endP1 = (self.endPoints[vertices[0]][0],self.endPoints[vertices[0]][1])
            endP2 = (self.endPoints[vertices[1]][0],self.endPoints[vertices[1]][1])
            endP3 = (self.endPoints[vertices[2]][0],self.endPoints[vertices[2]][1])
            destP1 = ((1-alpha) * sourceP1[0] + alpha * endP1[0],(1-alpha) * sourceP1[1] + alpha * endP1[1])
            destP2 = ((1-alpha) * sourceP2[0] + alpha * endP2[0],(1-alpha) * sourceP2[1] + alpha * endP2[1])
            destP3 = ((1-alpha) * sourceP3[0] + alpha * endP3[0],(1-alpha) * sourceP3[1] + alpha * endP3[1])
            sourceAffine = ColorAffine(np.array([sourceP1,sourceP2,sourceP3]),np.array([destP1,destP2,destP3]))
            sourceAffine.transform(self.startImage,sourceTarget)
            endAffine = ColorAffine(np.array([endP1,endP2,endP3]),np.array([destP1,destP2,destP3]))
            endAffine.transform(self.endImage,endTarget)
        return np.array((1-alpha) * sourceTarget +  alpha * endTarget,dtype=np.uint8)

    def generateMorphVideo(self,targetFolderPath, sequenceLength, includeReversed):
        if not os.path.isdir(targetFolderPath):
            os.makedirs(targetFolderPath)
        alphaValues = list(np.arange(0,1,1/(sequenceLength - 1)))
        alphaValues.append(1)
        if(includeReversed):
            alphaValues = list(alphaValues) + list(reversed(alphaValues)) 
        video = imageio.get_writer(targetFolderPath + '/morph.mp4',fps=5)
        frameCount = 1
        for alpha in alphaValues:
            blendImage = self.getBlendedImage(alpha)
            imageName = 'frame{:03d}.jpg'.format(frameCount)
            imagePath = path.join(targetFolderPath,imageName)
            imageio.imwrite(imagePath,blendImage)
            video.append_data(imageio.imread(imagePath))
            frameCount += 1
        video.close()

if __name__ == "__main__":

    # startimg = np.array(imageio.imread('WolfGray.jpg'),dtype=np.uint8)

    ### AFFINE CLASS TESTING ###    
    #destinationimg = np.zeros(startimg.shape,dtype=np.uint8)
    #inputMat = np.array([[200,500],[400,200],[600,500]],float)
    #destMat  = np.array([[200,500],[400,200],[600,500]],float)
    #testAffine = Affine(inputMat,destMat)
    #print(testAffine.matrix)
    #testAffine.transform(startimg,destinationimg)
    #imageio.imwrite('test.jpg',destinationimg)

    ### BLENDER CLASS TESTING ####

    
    # endimg = np.array(imageio.imread('Tiger2Gray.jpg'),dtype=np.uint8)
    startPoints = []
    with open('personalPointsStart.txt') as sourcefile:
        for line in sourcefile:
            m = re.search("(\d+)\s+?(\d+)",line)
            startPoints.append([m.group(1),m.group(2)])

    endPoints = []
    with open('personalPointsStartTarget.txt') as endfile:
        for line in endfile:
            m = re.search("(\d+)\s+?(\d+)",line)
            endPoints.append([m.group(1),m.group(2)])

    startPoints = np.array(startPoints,float)
    endPoints = np.array(endPoints,float)
    # testBlender = Blender(startimg,startPoints,endimg,endPoints)
    # finalImage = testBlender.getBlendedImage(0.5)
    # ##print(cProfile.run('finalImage = testBlender.getBlendedImage(0.6)'))
    # testBlender.generateMorphVideo('./gray_video',10,1)
    # imageio.imwrite('blenderTest.jpg',finalImage)

    colorstartimg = np.array(imageio.imread('WolfColor.jpg'),dtype=np.uint8)

    # COLOR AFFINE TESTING
    # colordestinationimg = np.zeros(colorstartimg.shape,dtype=np.uint8)
    # colorinputMat = np.array([[200,500],[400,200],[600,500]],float)
    # colordestMat  = np.array([[200,500],[400,200],[600,500]],float)
    # colortestAffine = ColorAffine(colorinputMat,colordestMat)
    # colortestAffine.transform(colorstartimg,colordestinationimg)
    # imageio.imwrite('testColor.jpg',colordestinationimg)
    
    # COLOR BLENDER TESTING
    # colorendimg = np.array(imageio.imread('Tiger2Color.jpg'),dtype=np.uint8)
    # colortestBlender = ColorBlender(colorstartimg,startPoints,colorendimg,endPoints)
    # colorfinalImage = colortestBlender.getBlendedImage(0.7)
    # # ##print(cProfile.run('finalImage = testBlender.getBlendedImage(0.6)'))
    # colortestBlender.generateMorphVideo('./color_video',40,1)
    # imageio.imwrite('blenderTest.jpg',colorfinalImage)

    ### PERSONAL MORPH SEQUENCE ###
    personalImage = np.array(imageio.imread('personalImage_resize.jpg'),dtype=np.uint8)
    endpersonalImage = np.array(imageio.imread('personalImageTarget_resize.jpg'),dtype=np.uint8)
    colortestPersonalBlender = ColorBlender(personalImage,startPoints,endpersonalImage,endPoints)
    colortestPersonalBlender.generateMorphVideo('./personal_video',50,1)

