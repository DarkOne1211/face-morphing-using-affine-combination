import dlib
import numpy as np
from skimage import io

predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = io.imread("personalImageTarget_resize.jpg")

dets = detector(img)

#output face landmark points inside retangle
#shape is points datatype
#http://dlib.net/python/#dlib.point
for k, d in enumerate(dets):
    shape = predictor(img, d)

vec = np.empty([73, 2], dtype = int)
for b in range(68):
    vec[b][0] = shape.part(b).x
    vec[b][1] = shape.part(b).y

vec[68] = np.array([0,0])
vec[69] = np.array([0,600])
vec[70] = np.array([600,0])
vec[71] = np.array([0,800])
vec[72] = np.array([800,0])

f = open("personalPointsStartTarget.txt","w+")
for points in vec:
    f.write(str(points[0]) + " " + str(points[1]) + "\n")