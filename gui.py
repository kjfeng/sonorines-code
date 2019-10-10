#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import numpy as np
import cv2
import imutils
import linvpy as lp
import math
import itertools
from sympy import symbols, solve

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRectF, QPointF, QSizeF, QPoint
from PyQt5.QtGui import QTransform, QBrush, QPainterPath, QPainter, QColor, QPen, QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QGraphicsRectItem, QApplication, QGraphicsView, QGraphicsScene, QGraphicsItem, QFileDialog, QDesktopWidget, QGraphicsPixmapItem, QMessageBox


# In[2]:


class DetectEllipse:
    def __init__(self, ballCoord, HLCoord, img):
        self.rois = []
        self.ball = ballCoord
        self.highlight = HLCoord
        ballCoord = np.array(ballCoord)
        self.ballRadius = ballCoord[:,2] # Extract r from ballCoord (x, y, r)
        self.ballCenter = ballCoord[:,:-1] # Extract (x, y) from ballCoord (x, y, r)
        self.img = img
        self.ellipses = []
        self.degrees = []
        self.centers_calib = []
        self.centers = []
        
        self.shadowDirect()
        #self.contour()
    
    #check direction of shadows to create rois
    def shadowDirect(self):
        i = 0    
        a,b,c,d = 0, 0, 0, 0
        for (x, y, r) in self.ball:
            #if the difference between the hl and the ball is '+' then the hl is below the ball's center
            if((self.highlight[i][1] - self.ball[i][1]) < 0 ): #y direction based on direction of highlight determine which side the shadow is on
                #print((self.highlight[i][1] - self.ball[i][1]))
                a, b = y+(3*r), y-r
                #print('DIFF:',a,b)
            else:
                b, a = y+(3*r), y-r
            #if the difference between the hl and the ball is '+' then the hl is to the right of the ball's center
            if((self.highlight[i][0] - self.ball[i][0]) > 0 ): # x direction based on direction of highlight determine which side the shadow is on
                c, d = x - (5*r), x+r
            else:
                d, c = x - (5*r), x+r
                
            #print('abcd: ',a,b,c,d)
            #bc of the coord system, the y increasing opposite from the norm (switch the y's in the range)
            #roi = self.img[int(a):int(b), int(c):int(d)]
            
            roi = self.img[int(y-r):int(y+(3*r)), int(x-(5*r)):int(x+r)]
            self.rois.append(roi)
            #cv2.imshow(str(i) + ' image', roi)
            i+=1
            #cv2.destroyAllWindows()
            
    def contour(self):
        #r = self.rois[3] #for testing
        numb = 0
        for r in self.rois: # for i in range(0,1) #for testing
            # Load the image, convert it to grayscale, blur it slightly, and threshold it
            #cv2.imshow('img1', r)
            gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('img2', gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            #cv2.imshow('img3', blurred)
            thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY_INV)[1] #inverse binary usually and 30
            #cv2.imshow('img4', thresh)

            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            cv2.drawContours(r, cnts, -1, (0,255,0), 2,maxLevel=0) # in green
            ##cv2.imshow('img', r)
            
            # Loop over the contours
            #print('len of contours', len(cnts))
            max_contour = cnts[0] #placeholder
            max_area = 0
            #find biggest contour
            for c in cnts:
                if c.shape[0] > 5:
                    area = cv2.contourArea(c)
                    if area > max_area:
                        max_area = area
                        max_contour = c

            ellipse = cv2.fitEllipse(max_contour)   
            (x, y), (MA, ma), angle = cv2.fitEllipse(max_contour)
            #print('CENTER: ', x, y)
            self.ellipses.append([x, y, MA, ma])
            self.degrees.append(angle)
            copyR = r.copy()
            cv2.ellipse(copyR,ellipse,(255,0,0,175),2) # in blue
            ##cv2.imshow(str(numb)+' ELLIPSE DRAWN:', copyR)         
            numb+=1
            
        #print('ellipses:',self.ellipses)
        print('degrees of rotation: ',self.degrees)
        points = self.get_center(self.ellipses)
        return points, self.degrees, self.centers
    
    def get_center(self, ellipses):
        count = 0
        for ix, iy, MA, ma in ellipses: #use the origin(top left corner) of the new roi to recalibrate
            #ix = ix - (5*self.ballRadius[count]) + self.ballCenter[count][0]
            #iy = iy - (2*self.ballRadius[count]) + self.ballCenter[count][1]
            ix = self.ballCenter[count][0] - (5*self.ballRadius[count]) + ix
            iy = self.ballCenter[count][1] - (self.ballRadius[count]) + iy
            self.centers_calib.append([ix, iy, MA, ma])
            self.centers.append([ix, iy])
            count += 1
        #print('CALIBRATED CENTERS', self.centers_calib)
        return self.centers_calib
    

              


# In[3]:


class Light:
    def __init__(self, ballCoord, HLCoord, img = None):
        self.rois = []
        self.ball = ballCoord
        #self.w = img.width()
        #self.h = img.height()
        self.errorBoolean = [False] * len(self.ball)
        #self.ball = [[701.2895833586797, 1091.7121093607425, 40.73242199381093],[1358.4276041603302, 434.2811197615678, 41.005078158267],[1358.42760416033, 762.5757812785145, 41.161718873562904],[1358.0, 1091.4242187214854, 40.57578127851457],[1687.4276041603302, 1091.14479167934, 41.0],[1029.0, 1092.141406240495, 40.85859375950491],[1687.8552083206598, 762.5740885590924, 40.73242199381093],[1687.0, 433.85859375950486, 40.858593759504856],[1030.2895833586797, 433.7205729578544, 41.1617188735633],[701.0, 435.0, 41.0],[1358.0, 1420.0, 41.0],[699.0, 1420.0, 41.0],[1029.1447916793395, 1420.9966145611556, 41.161718873563245],[1030.0, 763.0, 41.0],[1686.9966145611547, 1420.1380208016503, 41.16171887356313],[700.5723958396699, 762.7171875190098, 41.0]]
        self.highlight = []
        self.hlCoord = [[711.4386183168056, 757.7723729166229],[1701.7493091237686, 1417.2854741147453],[1041.5275926988065, 757.4724073011934],[1041.2922449924345, 1417.5388266247317],[711.1973793752408, 1417.592138125722],[1371.5065515618987, 1417.4934484381017],[711.8110370795226, 427.9055185397614],[1041.7499386441614, 427.46437262405243],[1702.574517479063, 427.1889629204773],[1702.3669346449392, 757.1802963436106],[1041.5745174790634, 1087.4717754014675],[1702.1441597796143, 1087.1447916793402],[1371.7628484998145, 1087.1870672213001],[1371.8879594458124, 757.4140624049516],[1372.4225260020633, 427.2760416033008],[711.2737169841529, 1087.5072800657379]]
        self.balls = []
        self.hlCoords = []
        self.ellipse_coord = []
        '''
        for (x, y, r) in self.ball:
            y = abs(self.h - y)
            ball = [x, y, r]
            self.balls.append(ball)
        for (x, y) in self.hlCoord:
            y = abs(self.h - y)
            coord = [x, y]
            self.hlCoords.append(coord)
        # count = 0
        print('REVERTED COORD HL and BALL: ', self.hlCoords,'\n', self.balls)
        '''
        #organize the highlight coord to correspond to the ball coord in index i, and calculate hz
        for cx, cy, r in self.ball:
            for hx, hy in HLCoord:#HLCoord
                if cx + r > hx > cx - r and cy + r > hy > cy - r:
                    coeff = [1, -2 * r, (hx - cx)**2 + (hy - cy)**2]
                    print("\nsphere coord:", [cx, cy, r])
                    print("highlight coord:", [hx, hy])
                    H = np.roots(coeff)
                    if 2 * r > H[0] > r:
                        self.highlight.append([hx, hy, H[0]])
                        print("hz is", H[0])
                    else:
                        self.highlight.append([hx, hy, H[1]])
                        print("hz is", H[1])

    
        self.degrees = []
        #find the degree of rotation of each ellipse that surrounds the shadows
        self.ellipse_pts, self.ellipse_degrees, self.centers = DetectEllipse(self.ball, HLCoord, img).contour()
        self.degrees = []
        for value in self.ellipse_degrees:
            degree = -1*(90 - value)
            self.degrees.append(degree)
        
        self.norm = self.normalize()
        self.detect_ellipse(self.norm, self.ball, self.highlight)
        self.lightPos = None
        #self.lightpos()
        
    #initially detect the ellipses that surround the shadows of the spheres 
    def detect_ellipse(self, normCoord, ballCoord, hlCoord):
        for i in range (0,len(ballCoord)):
            cx = ballCoord[i][0]
            cy = ballCoord[i][1]
            rad = ballCoord[i][2]
    
            hx = hlCoord[i][0]
            hy = hlCoord[i][1]
            hz = hlCoord[i][2]
            
            hl = abs(hz-rad) #highlight = dist btw hl and the position of the ball's center when viewed from z axis
            theta1 = math.acos(hl/rad)
            theta1 = math.degrees(theta1)
            
            #alternative linear algebra method of calculating theta
            N = normCoord[i]
            C = [0,0,1]
            theta = math.acos(np.dot(C, N))
            theta = math.degrees(theta)
            
            #check to see if different methods result in the same theta
            print('hl: ', hl, ' theta:', theta, ' <vs> ', theta1)
            phi = 2*theta
            phi = math.radians(phi)
            z = (2*rad)/math.cos(phi)
            z = abs(z)
    
            x = (rad + rad*math.cos(phi))/math.tan(phi)
            y = x + rad*math.sin(phi)
            
            print('X ',x, 'Y ', y, 'Z', z)
            self.ellipse_coord.append([cx-(z-rad), cy-rad, z, 2*rad]) #(top left coord, width, height) given #reverse signs if the shadow is to the right
    ####EDIT        
    def recal_hl_pos(self, shadow_width, radius):
        theta = (math.acos((2*radius) / shadow_width)) / 2
        hz = radius * math.cos(theta) + radius
        #theta = math.degrees(theta)
        #phi = 2* theta
        #phi = math.radians(phi) 
        #x = (1 + math.cos(phi))/math.tan(phi)
        #y = x + math.sin(phi)
        return hz
    
    def get_ellipse_centers(self):
        print('ELLIPSE COORDS',self.ellipse_coord)
        return self.ellipse_coord, self.degrees, self.highlight #self.ellipse_pts, self.ellipse_degrees, self.centers
        
    def alterCoordinateSystem(self, oldX, oldY):
        newRight, newLeft, newTop, newBottom = self.w, 0, self.h, 0
        oldRight, oldLeft, oldTop, oldBottom = self.w, 0, 0, self.h 
        
        newX = newLeft + ((oldX - oldLeft) / (oldRight - oldLeft)) * (newRight - newLeft)
        newY = newTop + ((oldY - oldTop) / (oldBottom - oldTop)) * (newBotom - newTop)
        return newX, newY
    # Find normal vectors from highlight relative to sphere
    def normalize(self):
        count = 0
        normal = []
        for cx, cy, r in self.ball:
            nx = (self.highlight[count][0] - cx) / r
            ny = (self.highlight[count][1] - cy) / r
            print("NX", nx, "\nNY", ny)
            #nz = math.sqrt(1 - (nx)**2 - (ny)**2)
            nz = (self.highlight[count][2] - r) / r
            #ex: NX 0.24511514751050714 NY -5.8729478316518575
            normal.append([nx, ny, nz])
            count += 1
        
        return normal
                       
    # Find two vectors (ui, vi) that are tangent to the sphere, intersecting at highlight point
    # Use ui, vi and highlight coordinates to find light position
    def lightpos(self, a = None, b = None):
        if a and b:
            A = a
            B = b
        else:
            A = []
            B = []
            count = 0
            for N in self.norm:
                u = np.cross(N, [1,0,0])
                v = np.cross(N, u)
                print("\nN", N)
                print("ui", u)
                print("vi", v)
                A.append(u)
                A.append(v)
                h = [self.highlight[count][0], self.highlight[count][1], self.highlight[count][2]]
                print("h", h)
                B.append([np.dot(u, h)])
                B.append([np.dot(v, h)])
                count += 1
        
        print("\nA", A)
        print("B", B)
        L = np.linalg.lstsq(A, B, rcond=None)[0]
        print("\nL", L)
        
        self.lightPos = L
        
        self.error(A, B)
    
    def error(self, A, B):
        max_ = 0
        errors = []
        count = 0
        for N in self.norm:
            ##Only if the norm was not 
            if self.errorBoolean[count] == False:
                u = np.cross(N, [1,0,0])
                v = np.cross(N, u)
                #print("\nN", N)
                #print("ui", u)
                #print("vi", v)

                lh = [self.lightPos[0]-self.highlight[count][0],self.lightPos[1]-self.highlight[count][1],self.lightPos[2]-self.highlight[count][2]]
                lh = [lh[0][0], lh[1][0], lh[2][0]]
                err = np.dot(lh, u) + np.dot(lh, v)

                print("l - h", lh)
                print("\nERROR:", err)

                errors.append(err)

                if abs(err) > max_:
                    max_ = err
            count += 1

        print("all the errors", errors)
        i = errors.index(max_)
        print("index of big error:", i)
        self.errorBoolean[i] = True 
        print("ERROR BOOLEAN LIST: ",self.errorBoolean)
        print("A before", A)
        print("B before", B)
        A.pop(2 * i)
        A.pop(2 * i)
        
        B.pop(2 * i)
        B.pop(2 * i)
        
        
        print("A after", A)
        print("B after", B)
        
        self.lightpos(A, B)
        print("\nL now", self.lightpos(A, B))
        
        
        
        


# In[4]:


class DetectHoughCircles:
    def __init__(self, image, width, height):
        self.image = cv2.imread(image) 
        self.circles = None # Array of circle coordinates in (x, y, r)
        self.width = width # Width of image in PyQt window
        self.height = height # Height of image in PyQt window
    
    def detect(self, radius = None):
        self.image = cv2.resize(self.image,(self.width,self.height))
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # Convert it to grayscale
        
        minDist = 250 #250
        minRadius = 20 #20
        maxRadius = 100
        param1 = 70
        param2 = 80
        
        # For redetection
        if radius:
            minRadius = int(radius - 10)
            maxRadius = int(radius + 10)

        # Detect circles in the image
        self.circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
        if self.circles is not None:
            length = len(self.circles[0])
        else:
            length = 0
               
        if self.circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            self.circles = np.round(self.circles[0, :]).astype("int")

        print("Final self.circles", self.circles)
        return self.circles


# In[5]:


class DetectHighlights:
    def __init__(self, roi, ballCoord):
        self.roi = roi # All cropped images of mirror ball
        self.centers = [] # Output of highlight centers
        
        ballCoord = np.array(ballCoord)
        self.ballRadius = ballCoord[:,2] # Extract r from ballCoord (x, y, r)
        self.ballCenter = ballCoord[:,:-1] # Extract (x, y) from ballCoord (x, y, r)
    
    # Compute center of the contour or blob with moments
    def center(self, c):
        M = cv2.moments(c)
        if(M["m00"] != 0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            return
        return [cX, cY]
    
    # Auto-detects closest highlight blob to center of each mirror ball
    def contour(self):
        self.centers = []
        count = 0
        for r in self.roi:
            # Load the image, convert it to grayscale, blur it slightly, and threshold it
            gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)[1]

            # Find contours in the thresholded image
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # Loop over the contours
            coord = []
            for c in cnts:
                # Set max and min size of contour to avoid anomaly
                #if len(c) > 4 and len(c) < 30: # Arbitrary numbers, need to change
                if len(c) > 4:
                    cX, cY = self.center(c)
                    if cX and cY != None:
                        coord.append([cX,cY])

            # Find the center closest to center of ball (highlight center)
            coord.sort(key = lambda P: (P[0]-self.ballRadius[count])**2 + (P[1]-self.ballRadius[count])**2)
            coord = coord[0]
            self.centers.append(coord)

            count += 1
            
        return self.get_centers()
        
    # Floodfills around given point to find blob of highlight
    def floodFill(self, x, y):
        print("FLOODFILLING!")
        count = 0
        roi_count = 0 # ROI index for moved point
        iroi = []
        for r in self.roi:
            rad = self.ballRadius[count]
            left = self.ballCenter[count][0] - rad
            right = self.ballCenter[count][0] + rad
            top = self.ballCenter[count][1] - rad
            bottom = self.ballCenter[count][1] + rad
            
            # Find which ROI contains moved point
            if left < x < right and top < y < bottom:
                iroi = r
                roi_count = count
                
            count += 1
        
        blurred = cv2.GaussianBlur(iroi, (5, 5), 0)
        thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]

        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8) # Size needs to be 2 pixels than image
        
        ix = x + self.ballRadius[roi_count] - self.ballCenter[roi_count][0]
        iy = y + self.ballRadius[roi_count] - self.ballCenter[roi_count][1]

        # Floodfill
        cv2.floodFill(im_floodfill, mask, (int(ix), int(iy)), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground
        blob = thresh | im_floodfill_inv
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
        blob_bw = cv2.threshold(blob, 250, 255, cv2.THRESH_BINARY_INV)[1]
        #blob_bw = cv2.bitwise_not(blob_bw)
        
        # Compute the center of the contour
        cX, cY = self.center(blob_bw)
        return self.get_centers(cX, cY, roi_count)
    
    # Recalibrate coordinates of highlight centers to full image
    def get_centers(self, x = None, y = None, roi_count = None):
        # For floodfill method
        if x and y and roi_count:
            cX = x - self.ballRadius[roi_count] + self.ballCenter[roi_count][0]
            cY = y - self.ballRadius[roi_count] + self.ballCenter[roi_count][1]
            return cX, cY
        # For contour method
        else:
            count = 0
            centers_calib = []
            for cX, cY in self.centers:
                cX = cX - self.ballRadius[count] + self.ballCenter[count][0]
                cY = cY - self.ballRadius[count] + self.ballCenter[count][1]
                count += 1
                centers_calib.append([cX, cY])
            return centers_calib
    
            


# In[6]:


class Ellipse(QGraphicsRectItem):
    #handleTopRight = 3
    handleMiddleLeft = 4
    handleMiddleRight = 5
    #handleBottomLeft = 6
    
    handleSize = +8.0
    handleSpace = -4.0

    # Changes cursor to signal operation they can perform on the shape
    # Handles are points where you can resize the shape
    handleCursors = {
        #handleTopRight: Qt.SizeBDiagCursor,
        handleMiddleLeft: Qt.SizeHorCursor,
        handleMiddleRight: Qt.SizeHorCursor,
        #handleBottomLeft: Qt.SizeBDiagCursor
    }

    def __init__(self, *args):
        # Initialize the circles and their handles
        super().__init__(*args)
        #super(Ellipse, self ).__init__( *args)
        self.handles = {}
        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.updateHandlesPos()
        
        self.angle = 0
        self.shift = False
        self.hover = False
        self.fineTune = False
        self.hx = 0
        self.hy = 0
        self.hz = 0
        self.scale = 0.1 # FURTHER FEATURE: have user input scale for fine-tune
    
    #sets the angle of the rotation of the ellipse
    def set_angle(self, angle, hx, hy):
        coord = self.coordinates()
        cx = coord[0]
        cy = coord[1]
        r = coord[2]
        self.angle = angle
        self.hx = hx
        self.hy = hy
        print('ANGLE: ', self.angle)
        self.setTransformOriginPoint(cx, cy)
        self.setRotation(self.angle)
        
        self.hl_pos = Dot(hx-1, hy-1, 2,2) # indicates location of the highlight
        
    #returns highlight position    
    def hl_pos(self):
        return self.hl_pos
    
    #decreases the angle the ellipse is rotated
    def decrease_angle(self):
        self.angle -=2
        self.setRotation(self.angle)
        
    #increases the angle the ellipse is rotated
    def increase_angle(self):
        self.angle +=2
        self.setRotation(self.angle)
        
    # Returns the resize handle below the given point
    def handleAt(self, point):
        for k, v, in self.handles.items():
            if v.contains(point):
                return k
        return None
    
    # Executed when the mouse moves over the shape (NOT PRESSED)
    def hoverMoveEvent(self, moveEvent):
        self.hover = True
        if self.isSelected():
            handle = self.handleAt(moveEvent.pos())
            cursor = Qt.ArrowCursor if handle is None else self.handleCursors[handle]
            self.setCursor(cursor)
        super().hoverMoveEvent(moveEvent)

    # Executed when the mouse leaves the shape (NOT PRESSED)
    def hoverLeaveEvent(self, moveEvent):
        self.hover = False
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(moveEvent)

    # Executed when the mouse is pressed on the item
    def mousePressEvent(self, mouseEvent):
        self.handleSelected = self.handleAt(mouseEvent.pos())
        self.mousePressPos = mouseEvent.pos()
        self.mousePressRect = self.boundingRect()
        super().mousePressEvent(mouseEvent)

    # Executed when the mouse is being moved over the item while being pressed
    def mouseMoveEvent(self, mouseEvent):
        if self.handleSelected is not None:
            self.interactiveResize(mouseEvent.pos())
        #elif self.isSelected():
        #    self.dragImage(mouseEvent)
        else:
            super().mouseMoveEvent(mouseEvent)

    # Executed when the mouse is released from the item
    def mouseReleaseEvent(self, mouseEvent):
        super().mouseReleaseEvent(mouseEvent)
        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None
        self.update()
            
    # Returns the bounding rect of the shape (including the resize handles)
    def boundingRect(self):
        o = self.handleSize + self.handleSpace
        return self.rect().adjusted(-o, -o, o, o)

    # Update current resize handles according to the shape size and position
    def updateHandlesPos(self):
        s = self.handleSize
        b = self.boundingRect()
        #self.handles[self.handleTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.handles[self.handleMiddleLeft] = QRectF(b.left(), b.center().y() - s / 2, s, s)
        self.handles[self.handleMiddleRight] = QRectF(b.right() - s, b.center().y() - s / 2, s, s)
        #self.handles[self.handleBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)

    # Perform shape interactive resize
    def interactiveResize(self, mousePos):
        offset = self.handleSize + self.handleSpace
        boundingRect = self.boundingRect()
        rect = self.rect()
        degree = 0
        diff = QPointF(0, 0) # Center of circle coordinates
        transform = QTransform()
        self.prepareGeometryChange()
        

        # Middle left adjustments
        if self.handleSelected == self.handleMiddleLeft:
            fromX = self.mousePressRect.left()
            toX = fromX + mousePos.x() - self.mousePressPos.x()
            diff.setX(toX - fromX)
            boundingRect.setLeft(toX)
            rect.setLeft(boundingRect.left() + offset)
            self.setRect(rect)
            self.shift = True

        elif self.handleSelected == self.handleMiddleRight:
            fromX = self.mousePressRect.right()
            toX = fromX + mousePos.x() - self.mousePressPos.x()
            diff.setX(toX - fromX)
            boundingRect.setRight(toX)
            rect.setRight(boundingRect.right() - offset)
            self.setRect(rect)
            self.shift = True
            
        self.updateHandlesPos()
        if self.shift == True:
            width = rect.width()
         ## do math to take in width and produce new hl pos 
            ##and change the position of the hl_pos but make sure the user cant move it
            
    # Returns coordinates of ellipse
    def coordinates(self):
        r = self.rect().width() / 2 #width
        x = self.rect().x() + r
        y = self.rect().y() + r
        return [x, y, r]
    def shadow_point(self): #assume that the leftmost point of the rect is the shadow's x value
        x = self.rect().left()
        w = self.rect().width()
        y = math.sqrt((w**2)-(x**2))
        return [x, y]
    # Returns the shape of this item as a QPainterPath in local coordinates
    def shape(self):
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for shape in self.handles.values():
                path.addEllipse(shape)
        return path

    def paint(self, painter, option, widget=None):
        # Paint the circle region in the graphic view
        painter.setBrush(QBrush(QColor(255, 0, 0, 0)))
        painter.setPen(QPen(QColor(255, 0, 0, 255), 0.0, Qt.SolidLine))
        painter.drawEllipse(self.rect()) # Draw ellipse in bounding rectangle
        ## draw the circle of the highlight position
        
        painter.setRenderHint(QPainter.Antialiasing)
        # Color and thickness of handle nodes
        if self.hover is False:
            painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
        painter.setPen(QPen(QColor(0, 0, 0, 255), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for handle, rect in self.handles.items():
            if self.handleSelected is None or handle == self.handleSelected:
                painter.drawEllipse(rect)


# In[7]:


class Ball(QGraphicsRectItem):
    handleTopMiddle = 2
    handleMiddleLeft = 4
    handleMiddleRight = 5
    handleBottomMiddle = 7

    handleSize = +8.0
    handleSpace = -4.0

    # Changes cursor to signal operation they can perform on the shape
    # Handles are points where you can resize the shape
    handleCursors = {
        handleTopMiddle: Qt.SizeVerCursor,
        handleMiddleLeft: Qt.SizeHorCursor,
        handleMiddleRight: Qt.SizeHorCursor,
        handleBottomMiddle: Qt.SizeVerCursor,
    }

    def __init__(self, *args):
        # Initialize the circles and their handles
        super().__init__(*args)
        self.handles = {}
        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.updateHandlesPos()
        
        self.shift = False
        self.hover = False
        self.fineTune = False
        self.scale = 0.1 # FURTHER FEATURE: have user input scale for fine-tune
        
    # Returns the resize handle below the given point
    def handleAt(self, point):
        for k, v, in self.handles.items():
            if v.contains(point):
                return k
        return None
    
    #def shiftPressed(self, pos):
        #print("signal received")
        #not self.fineTune
    
    # Executed when the mouse moves over the shape (NOT PRESSED)
    def hoverMoveEvent(self, moveEvent):
        self.hover = True
        if self.isSelected():
            handle = self.handleAt(moveEvent.pos())
            cursor = Qt.ArrowCursor if handle is None else self.handleCursors[handle]
            self.setCursor(cursor)
        super().hoverMoveEvent(moveEvent)

    # Executed when the mouse leaves the shape (NOT PRESSED)
    def hoverLeaveEvent(self, moveEvent):
        self.hover = False
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(moveEvent)

    # Executed when the mouse is pressed on the item
    def mousePressEvent(self, mouseEvent):
        self.handleSelected = self.handleAt(mouseEvent.pos())
        self.mousePressPos = mouseEvent.pos()
        self.mousePressRect = self.boundingRect()
        super().mousePressEvent(mouseEvent)

    # Executed when the mouse is being moved over the item while being pressed
    def mouseMoveEvent(self, mouseEvent):
        if self.handleSelected is not None:
            self.interactiveResize(mouseEvent.pos())
        elif self.isSelected():
            self.dragImage(mouseEvent)
        else:
            super().mouseMoveEvent(mouseEvent)

    # Executed when the mouse is released from the item
    def mouseReleaseEvent(self, mouseEvent):
        super().mouseReleaseEvent(mouseEvent)
        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None
        self.update()

    # Returns the bounding rect of the shape (including the resize handles)
    def boundingRect(self):
        o = self.handleSize + self.handleSpace
        return self.rect().adjusted(-o, -o, o, o)
    
    def dragImage(self, mouseEvent):
        boundingRect = self.boundingRect()
        rect = self.rect()
        offset = self.handleSize + self.handleSpace
        
        fromX = self.mousePressRect.left()
        fromY = self.mousePressRect.top()
        toX = fromX + mouseEvent.pos().x() - self.mousePressPos.x()
        toY = fromY + mouseEvent.pos().y() - self.mousePressPos.y()
        
        boundingRect.setLeft(toX)
        boundingRect.setRight(toX + self.mousePressRect.width())
        boundingRect.setTop(toY)
        boundingRect.setBottom(toY + self.mousePressRect.width())
        
        rect.setBottom(boundingRect.bottom() - offset)  
        rect.setTop(boundingRect.top() + offset)
        rect.setRight(boundingRect.right() - offset)
        rect.setLeft(boundingRect.left() + offset)
        
        self.shift = True
        self.setRect(rect)
        
        self.updateHandlesPos()

    # Update current resize handles according to the shape size and position
    def updateHandlesPos(self):
        s = self.handleSize
        b = self.boundingRect()
        self.handles[self.handleTopMiddle] = QRectF(b.center().x() - s / 2, b.top(), s, s)
        self.handles[self.handleMiddleLeft] = QRectF(b.left(), b.center().y() - s / 2, s, s)
        self.handles[self.handleMiddleRight] = QRectF(b.right() - s, b.center().y() - s / 2, s, s)
        self.handles[self.handleBottomMiddle] = QRectF(b.center().x() - s / 2, b.bottom() - s, s, s)

    # Perform shape interactive resize (fixed circle)
    def interactiveResize(self, mousePos):
        offset = self.handleSize + self.handleSpace
        boundingRect = self.boundingRect()
        rect = self.rect()
        diff = QPointF(0, 0) # Center of circle coordinates

        self.prepareGeometryChange()

        # Top middle adjustments
        if self.handleSelected == self.handleTopMiddle:
            diffY = mousePos.y() - self.mousePressPos.y()
            if self.fineTune is True:
                diffY *= self.scale
            
            fromY = self.mousePressRect.top()
            toY = fromY + diffY
            
            fromXleft = self.mousePressRect.left()
            fromXright = self.mousePressRect.right()
            toXleft = fromXleft + diffY / 2
            toXright = fromXright - diffY / 2
            
            average = diffY / 2
            diff.setY(average)
            
            boundingRect.setTop(toY + average)
            boundingRect.setLeft(toXleft + (average / 2))
            boundingRect.setRight(toXright - (average / 2))
            
            rect.setTop(boundingRect.top() + offset)
            rect.setRight(boundingRect.right() - offset)
            rect.setLeft(boundingRect.left() + offset)
            
            self.shift = True
            self.setRect(rect)

        # Middle left adjustments
        elif self.handleSelected == self.handleMiddleLeft:
            diffX = mousePos.x() - self.mousePressPos.x()
            if self.fineTune is True:
                diffX *= self.scale
            
            fromX = self.mousePressRect.left()
            toX = fromX + diffX
            
            fromYtop = self.mousePressRect.top()
            fromYbottom = self.mousePressRect.bottom()
            
            toYtop = fromYtop + diffX / 2
            toYbottom = fromYbottom - diffX / 2
            
            average = diffX / 2
            diff.setX(average)
            
            boundingRect.setLeft(toX + average)
            boundingRect.setTop(toYtop + (average / 2))
            boundingRect.setBottom(toYbottom - (average / 2))
            
            rect.setLeft(boundingRect.left() + offset)
            rect.setTop(boundingRect.top() + offset)
            rect.setBottom(boundingRect.bottom() - offset)
            
            self.shift = True
            self.setRect(rect)

        # Middle right adjustments
        elif self.handleSelected == self.handleMiddleRight:
            diffX = mousePos.x() - self.mousePressPos.x()
            if self.fineTune is True:
                diffX *= self.scale
            
            fromX = self.mousePressRect.right()
            toX = fromX + diffX
            
            fromYtop = self.mousePressRect.top()
            fromYbottom = self.mousePressRect.bottom()
            
            toYtop = fromYtop - diffX / 2
            toYbottom = fromYbottom + diffX / 2

            average = diffX / 2
            diff.setX(average)
            
            boundingRect.setRight(toX + average)
            boundingRect.setTop(toYtop - (average / 2))
            boundingRect.setBottom(toYbottom + (average / 2))
            
            rect.setRight(boundingRect.right() - offset)
            rect.setTop(boundingRect.top() + offset)
            rect.setBottom(boundingRect.bottom() - offset)
            
            self.shift = True
            self.setRect(rect)

        # Bottom middle adjustments
        elif self.handleSelected == self.handleBottomMiddle:
            diffY = mousePos.y() - self.mousePressPos.y()
            if self.fineTune is True:
                diffY *= self.scale
            
            fromY = self.mousePressRect.bottom()
            toY = fromY + diffY
            
            fromXleft = self.mousePressRect.left()
            fromXright = self.mousePressRect.right()
            
            toXleft = fromXleft - diffY / 2
            toXright = fromXright + diffY / 2
            
            average = diffY / 2
            diff.setY(average)
            
            boundingRect.setBottom(toY + average)
            boundingRect.setLeft(toXleft - (average / 2))
            boundingRect.setRight(toXright + (average / 2))
            
            rect.setBottom(boundingRect.bottom() - offset)
            rect.setRight(boundingRect.right() - offset)
            rect.setLeft(boundingRect.left() + offset)
            
            self.shift = True
            self.setRect(rect)

        self.updateHandlesPos()

    # Returns coordinates of circle
    def coordinates(self):
        r = self.rect().width() / 2
        x = self.rect().x() + r
        y = self.rect().y() + r
        return [x, y, r]
    
    # Returns the shape of this item as a QPainterPath in local coordinates
    def shape(self):
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for shape in self.handles.values():
                path.addEllipse(shape)
        return path

    def paint(self, painter, option, widget=None):
        # Paint the circle region in the graphic view
        painter.setBrush(QBrush(QColor(255, 0, 0, 0)))
        painter.setPen(QPen(QColor(255, 0, 0, 255), 0.0, Qt.SolidLine))
        painter.drawEllipse(self.rect()) # Draw ellipse in bounding rectangle
        
        painter.setRenderHint(QPainter.Antialiasing)
        # Color and thickness of handle nodes
        if self.hover is False:
            painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
        painter.setPen(QPen(QColor(0, 0, 0, 255), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for handle, rect in self.handles.items():
            if self.handleSelected is None or handle == self.handleSelected:
                painter.drawEllipse(rect)


# In[8]:


class Dot(QGraphicsRectItem):

    def __init__(self, *args):
        # Initialize the circles and their handles
        super().__init__(*args)
        self.mousePressPos = None
        self.mousePressRect = None
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        #self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.shift = False
        
    # Executed when the mouse is pressed on the item
    def mousePressEvent(self, mouseEvent):
        self.mousePressPos = mouseEvent.pos()
        self.mousePressRect = self.boundingRect()
        super().mousePressEvent(mouseEvent)
        
    def mouseMoveEvent(self, mouseEvent):
        if self.isSelected():
            self.dragImage(mouseEvent)
        else:
            super().mouseMoveEvent(mouseEvent)
    
    # Executed when the mouse is released from the item
    def mouseReleaseEvent(self, mouseEvent):
        super().mouseReleaseEvent(mouseEvent)
        self.mousePressPos = None
        self.mousePressRect = None
        self.update()

    def dragImage(self, mouseEvent):
        boundingRect = self.boundingRect()
        rect = self.rect()
        
        fromX = self.mousePressRect.left()
        fromY = self.mousePressRect.top()
        toX = fromX + mouseEvent.pos().x() - self.mousePressPos.x()
        toY = fromY + mouseEvent.pos().y() - self.mousePressPos.y()
        
        boundingRect.setLeft(toX)
        boundingRect.setRight(toX + 2)
        boundingRect.setTop(toY)
        boundingRect.setBottom(toY + 2)
            
        rect.setRight(boundingRect.right())
        rect.setTop(boundingRect.top())
        rect.setBottom(boundingRect.bottom())  
        rect.setLeft(boundingRect.left())  
        
        self.shift = True
        self.setRect(rect)
    def shifted(self):
        if self.shift == True:
            return 10
        else:
            return -10
    # Returns coordinates of dot
    def coordinates(self):
        r = self.boundingRect().width() / 2
        x = self.boundingRect().x() + r
        y = self.boundingRect().y() + r
        return [x, y]
    
    # Returns the bounding rect of the shape
    def boundingRect(self):
        return self.rect()

    # Returns the shape of this item as a QPainterPath in local coordinates
    def shape(self):
        path = QPainterPath()
        path.addRect(self.rect())
        return path

    def paint(self, painter, option, widget=None):
        # Paint the circle region in the graphic view
        painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
        painter.setPen(QPen(QColor(255, 0, 0, 255), 0.0, Qt.SolidLine))
        painter.drawEllipse(self.rect()) # Draw ellipse in bounding rectangle
        
        painter.setRenderHint(QPainter.Antialiasing)


# In[9]:


class PhotoViewer(QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)
    #shiftClicked = QtCore.pyqtSignal(QtCore.QPoint)
    

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self) # where the photo and items are located in
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self._x = 0
        self._y = 0
        self.fineTune = False
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # Scroll bars (on or off)
        #self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        #self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
    
    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    # Key Events for zooming in/out image + deleting items
    def keyPressEvent(self, event):
        deviceTransform = QTransform(1, 0, 0, 1, 1.0, 1.0)
        factor = 1
        if self.hasPhoto(): 
            # Deletes item if selected and pressed 'backspace'
            if event.key() == Qt.Key_Backspace:
                location = self._scene.itemAt(self._x,self._y, deviceTransform)
                print('Location of deleted item: ', self._x, self._y)
                if location != self._photo:
                    self._scene.removeItem(location)
                    
            #if event.key() == Qt.Key_Shift:
                #self.fineTune = not self.fineTune
                #print("shift is pressed")
                #print(self.fineTune)
                
            #Allows the user to shift the degree of rotation that the ellipse item is oriented
            item = self._scene.itemAt(self._x,self._y, deviceTransform)
            if event.key() == Qt.Key_Left:
                if item != self._photo:
                    item.decrease_angle()
                
            elif event.key() == Qt.Key_Right:
                if item != self._photo:
                    item.increase_angle()
                
            # Zooming, 'up' arrow for zoom in and 'down' arrow to zoom out    
            if event.key() == Qt.Key_Up:
                factor = 1.2
                self._zoom += 1
            elif event.key() == Qt.Key_Down:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0
   
    def mousePressEvent(self, event):
        p = QPointF(0,0)
        if self._photo.isUnderMouse():
            self.photoClicked.emit(QtCore.QPoint(event.pos()))
            p = self.mapToScene(event.x(), event.y())
            self._x = p.x()
            self._y = p.y()
        super(PhotoViewer, self).mousePressEvent(event)
    


# In[10]:


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = PhotoViewer(self)
        self.mirror_img = None
        self.img_filename = []
        self.output = None # Saved file of coordinates and light positions
        self.img_width = None
        self.img_height = None
        self.ballCoord = None
        self.highlightCoord = []
        self.rois = []
        self.initUI()
        self.lx = 0
        self.ly = 0
        self.lz = 0
        
    def initUI(self):
        # 'Load image' button
        self.btnLoad = QtWidgets.QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
                
        # Button to change from drag/pan to getting pixel info
        self.btnBallInfo = QtWidgets.QToolButton(self)
        self.btnBallInfo.setText('Indicate location of mirror balls')
        self.btnBallInfo.clicked.connect(self.ballInfo)
        
        # Button to save ball center and radius after user adjustment
        self.btnBallCoord = QtWidgets.QToolButton(self)
        self.btnBallCoord.setText('Save mirror ball coord')
        self.btnBallCoord.clicked.connect(self.getBallCoord)
        
        # Button to change from drag/pan to getting pixel info
        self.btnHLInfo = QtWidgets.QToolButton(self)
        self.btnHLInfo.setText('Indicate location of highlights')
        self.btnHLInfo.clicked.connect(self.highlightInfo)
        
        # Button to save highlight center after user adjustment
        self.btnHLCoord = QtWidgets.QToolButton(self)
        self.btnHLCoord.setText('Save highlight coord')
        self.btnHLCoord.clicked.connect(self.getHighlightCoord)
        
        # Button to calculate light position with ball and highlight coord
        self.btnLightPos = QtWidgets.QToolButton(self)
        self.btnLightPos.setText('Calculate light position')
        self.btnLightPos.clicked.connect(self.getLightPos)
        
        #recalculate using saved ellipse coord
        self.btnLightPosCoord = QtWidgets.QToolButton(self)
        self.btnLightPosCoord.setText('Recalculate light position')
        self.btnLightPosCoord.clicked.connect(self.getLightPosCoord)
        
        # Text box to guide the user along the GUI
        self.guiInfo = QtWidgets.QLineEdit(self)
        self.guiInfo.setReadOnly(True)
        self.guiInfo.setText('Welcome to the Mirror Ball GUI. Press Load Image to start.')
        #self.viewer.photoClicked.connect(self.photoClicked)
        
        # Arrange layout
        # Top row of buttons
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnBallInfo)
        HBlayout.addWidget(self.btnBallCoord)
        HBlayout.addWidget(self.btnHLInfo)
        HBlayout.addWidget(self.btnHLCoord)
        HBlayout.addWidget(self.btnLightPos)
        HBlayout.addWidget(self.btnLightPosCoord)
        
        VBlayout.addLayout(HBlayout)
        
        # Bottom row of text/instructions
        HB2layout = QtWidgets.QHBoxLayout()
        HB2layout.setAlignment(QtCore.Qt.AlignLeft)
        HB2layout.addWidget(self.guiInfo)
        
        VBlayout.addLayout(HB2layout)
        
    def loadImage(self):
        # Check to see if already existing image
        if self.btnLoad.text() == "Load new image":
            for item in self.viewer._scene.items():
                if item != self.viewer._photo:
                    self.viewer._scene.removeItem(item)
            self.btnBallInfo.setText("Indicate location of mirror balls")
            self.btnBallInfo.repaint()
            self.btnHLInfo.setText("Indicate location of highlights")
            self.btnHLInfo.repaint()
        
        self.img_filename = QFileDialog.getOpenFileName(None, "Open Image")
        self.mirror_img = QPixmap.fromImage(QImage(self.img_filename[0]))
        self.viewer.setPhoto(self.mirror_img)
        self.img_width = self.mirror_img.size().width()
        self.img_height = self.mirror_img.size().height()
        
        self.guiInfo.setText('Now indicate the location of the mirror ball. Adjust size with handles and zoom with arrow keys. If auto-detection is inaccurate, adjust one or more circles and re-press button. Click save when you are done.')
        self.btnLoad.setText('Load new image')
        self.btnLoad.repaint()
        
    # Run Hough Circle Detection and add resizable circle at every mirror ball
    def ballInfo(self):
        # Check to see if the balls have already been detected
        if self.btnBallInfo.text() == "Refresh detection":
            self.redetectBallInfo()
            print("BALL INFO BUTTON PRESSED")
            return
        
        mirror_circle = []
        mirror_circles = DetectHoughCircles(self.img_filename[0], self.img_width, self.img_height).detect()
        for (x, y, r) in mirror_circles:
            item = Ball(x - r, y - r, 2*r, 2*r)
            #item2 = Ellipse(x - r, y - r, 2*r, 2*r)
            self.viewer._scene.addItem(item)
            
            
        self.btnBallInfo.setText("Refresh detection")
        self.btnBallInfo.repaint()
    
    #redetects the outlines of the mirrored spheres in case not all were detected
    def redetectBallInfo(self):
        r = 0
        print("REDETECTION WAS SELECTED")
        new_coord = [] # New ball coordinates to display
        fixed_coord = [] # Adjusted ball coordinates
        old_coord = self.getBallCoord() # Old ball coordinates
        for item in self.viewer._scene.items():
            if item != self.viewer._photo:
                if item.shift == True: #if the item moved
                    r += 1 # if item is shifted then redetect circles
                    fixed_coord.append(item.coordinates())# appends coordinates of ball to fixed coord
                self.viewer._scene.removeItem(item)
    
        if r != 0:
            redetected_circles = DetectHoughCircles(self.img_filename[0], self.img_width, self.img_height).detect(r)
            print("R is not equal to 0")
            # If redetecting using Hough Circle works, add coordinates to mirror_circles
            if redetected_circles is not None and len(redetected_circles) > 17: #>16?
                for (_x, _y, _r) in redetected_circles:
                    for (x, y, r) in fixed_coord:
                        if not x + r > _x > x - r and y + r > _y > y - r:
                            item = Ball(_x - _r, _y - _r, 2 * _r, 2 * _r)
                            self.viewer._scene.addItem(item)
                            
                for (x, y, r) in fixed_coord:
                    item = Ball(x - r, y - r, 2*r, 2*r)
                    self.viewer._scene.addItem(item)

            # If not, manually shift all circles across and below adjusted sphere
            else:
                # Changing all coordinates of circles on the same row/col as adjusted
                for (_x, _y, _r) in old_coord:
                    for (x, y, r) in fixed_coord:
                        if x + 2 * r > _x > x - 2 * r:
                            new_coord.append([x, _y, r])
                        elif y + 2 * r > _y > y - 2 * r:
                            new_coord.append([_x, y, r])
                        else:
                            new_coord.append([_x, _y, r])
                                
                # Catch circles that are duplicates
                new_coord = [list(i) for i in set(map(tuple, new_coord))]
                
                # Catch circles that are too close to each other
                # Iterates through every unique combination of coordinates
                for (x, y, r),(_x, _y, _r) in itertools.combinations(new_coord, 2):
                    if math.sqrt(((x-_x)**2)+((y-_y)**2)) < (2 * r):
                        try:
                            new_coord.remove([x, y, r])
                        except ValueError:
                            pass
                
                for (x, y, r) in new_coord:
                    item = Ball(x - r, y - r, 2*r, 2*r)
                    self.viewer._scene.addItem(item)
                    
        # If pressed button by mistake
        else:
            print("BUTTON PRESSED BY MISTAKE")
            for (x, y, r) in old_coord:
                item = Ball(x - r, y - r, 2*r, 2*r)
                self.viewer._scene.addItem(item)
    
    # Returns mirror ball center coordinates and radius (x, y, r)
    def getBallCoord(self):
        self.ballCoord = []
        for item in self.viewer._scene.items():
            if (item.boundingRect().x() != 0):
                self.ballCoord.append(item.coordinates())
        
        if self.btnBallInfo.text() != "Refresh detection":
            self.guiInfo.setText("Saved! Now indicate the location of the highlight coordinates. If auto-detection is inaccurate, move dot into highlight and re-press button. Click save when done.")
            self.guiInfo.repaint()
                
        return self.ballCoord

    # Runs Highlight Detection to get the center coord of each highlight
    def highlightInfo(self):
        # Check to see if the highlights have already been detected
        if self.btnHLInfo.text() == "Refresh detection":
            self.redetectHLInfo()
            return
       
        for item in self.viewer._scene.items():
            if item != self.viewer._photo:
                self.viewer._scene.removeItem(item) 
        
        self.rois = []
        image = cv2.imread(self.img_filename[0])
        image = cv2.bitwise_not(image) # Invert color (better for contour detection)
        
        for (x, y, r) in self.ballCoord:
            roi = image[int(y - r):int(y + r), int(x - r):int(x + r)]
            self.rois.append(roi)
            
        highlight_center = DetectHighlights(self.rois, self.ballCoord).contour()
        for (x, y) in highlight_center:
            item = Dot(x-1, y-1, 2, 2)
            self.viewer._scene.addItem(item)
            
        self.btnHLInfo.setText("Refresh detection")
        self.btnHLInfo.repaint()
     
    #Checks to see which highlights are shifted and redects the center of the highlights using floodfill method
    def redetectHLInfo(self):
        print("REDETECT HIGHLIGHTS")
        for item in self.viewer._scene.items():
            if item != self.viewer._photo:
                #sX, sY = item.shift()
                if item.shifted() > 0:
                    cx, cy = item.coordinates()
                    self.viewer._scene.removeItem(item)
                    x, y = DetectHighlights(self.rois, self.ballCoord).floodFill(cx, cy)
                    #print("X and Y", x, y)
                    self.viewer._scene.addItem(Dot(x, y, 2, 2))
    
    # Returns highlight center coordinates (x, y)
    def getHighlightCoord(self):
        self.highlightCoord = []
        for item in self.viewer._scene.items():
            if (item.boundingRect().x() != 0):
                self.highlightCoord.append(item.coordinates())
        
        self.guiInfo.setText("Done! Now you can calculate the position of light.")
        self.guiInfo.repaint()
                
        return self.highlightCoord
    
    # Returns position of light from mirror ball and highlight coordinates
    def getLightPos(self):
        for item in self.viewer._scene.items():
            if item != self.viewer._photo:
                self.viewer._scene.removeItem(item) 
                
        image = cv2.imread(self.img_filename[0])
        light = Light(self.ballCoord, self.highlightCoord,image)
        #image = cv2.bitwise_not(image)
        
        points, angles, highlight_pos = light.get_ellipse_centers()
        
        i = 0
        for [x, y, MA, ma] in points:
            angle = angles[i]
            hx, hy = highlight_pos[i][0], highlight_pos[i][1]
            print('\nLTopLeft Point',x, y )
            #hl = item.hl_pos()
            #self.viewer._scene.addItem(hl)
            item = Ellipse(x, y, MA, ma)
            x1, y1= item.coordinates()[0], item.coordinates()[1]
            print('\nCenter Point', x1, y1)
            self.viewer._scene.addItem(item)
            item.set_angle(angle, hx, hy)
            i+=1
        '''    
        for item in self.viewer._scene.items():
            if item != self.viewer._photo:
                if item.shift == True:
                    width = item.width()
                    radius = item.height()/2
                    hx, hy = light.recal_hl_pos(width, radius)
                    hl = Dot(hx-1, hy-1, 2, 2)
                    self.viewer._scene.addItem(hl)
                    ##recalculate highlight position
        '''
        #return self.lx, self.ly, self.lz
        
    def getLightPosCoord(self):
        self.ellipseCoord = []
        for item in self.viewer._scene.items():
            if (item.boundingRect().x() != 0):
                self.ellipseCoord.append(item.shadow_point())
                
        self.ellipseCoord()
        
        #self.lx, self.ly, self.lz = light.lightpos()
        #self.guiInfo.setText('The light direction is: <%.5f, %.5f, %.5f>' % (self.lx, self.ly, self.lz))
        #self.guiInfo.repaint()
        
    def closeEvent(self, event):
        if self.img_filename[0] is not []:
            close = QMessageBox()
            img_name = self.img_filename[0].rsplit('/', 1)
            close.setText("Save the coordinates of " + img_name[-1] + " ?")
            close.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            close = close.exec()

            if close == QMessageBox.Yes:
                output_filename = QFileDialog.getSaveFileName(None, "Save File","","Text Files (*.txt)")
                output = open(output_filename[0], 'w',)
                outstring = ""
                
                if self.ballCoord is not None:
                    outstring += "Sphere Coordinates: \n"
                    for i in self.ballCoord:
                        outstring += ("\n" + str(i))
                        
                if self.highlightCoord is not None:
                    outstring += "\nHighlight Coordinates: \n"
                    for i in self.highlightCoord:
                        outstring += ("\n" + str(i))
                        
                if self.getLightPos is not None:
                    l = [self.lx, self.ly, self.lz]
                    outstring += "\nLight Position/Direction: \n"
                    outstring += str(l)
                
                output.write(outstring)
                output.close()
                
                event.accept()
            else:
                event.accept()
        else:
            event.accept()
    
    #def photoClicked(self, pos):
        #if self.viewer.dragMode()  == QGraphicsView.NoDrag:
            #self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))


# In[ ]:


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    output_file = Window().output

    screen = QDesktopWidget().screenGeometry()
    window.setGeometry(0, 0, screen.width(), screen.height())
    
    window.show()
    sys.exit(app.exec_())

