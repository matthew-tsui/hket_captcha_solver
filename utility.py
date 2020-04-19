# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
from os import listdir
import imutils

class utility:
    
   def resize_image(self, image, width, height):

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image

    def segment_image(image):

        #convet to black and gray
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 

        # remove the white background
        ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)  

        # find the margins of each character
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        image_boxes = []
        letter_boxes = []
        cache = ()

        # loop over each margin
        for contour in contours:

            # use a rectangle to bound the margin
            (x, y, w, h) = cv2.boundingRect(contour)

            #print(w * h, (x, y, w, h))
            
            # add only external border box
            # for w < 4 and h < 6, there are the inner boxes that should be excluded
            if w > 4 and h > 6 and x != 0 and y != 0:
                if (w * h) > 95:
                    if w > h:
                        half_width = int(w / 2)
                        letter_boxes.append((x, y, half_width, h))
                        letter_boxes.append((x + half_width, y, half_width, h))
                    else:
                        letter_boxes.append((x, y, w, h))
                elif w * h > 60:
                    # if area > 6, it may be a part of 5
                    if len(cache):
                        # get part of the 5 and merge with another part
                        x1, y1, w1, h1 = cache
                        letter_boxes.append((min(x,x1), min(y,y1), 10, 12))
                        cache = ()
                    else:
                        cache = (x, y, w, h)

        # sort by x coordinates so that it is from left to right
        letter_boxes = sorted(letter_boxes, key=lambda x: x[0])

        # if we could not split 6 digits, skip this sample
        if len(letter_boxes) != 6:
            print(len(letter_boxes))
            return

        for i,letter_box in enumerate(letter_boxes):
            
            x, y, w, h = letter_box

            letter_image = binary[y:y+h,x:x+w]

            image_boxes.append(letter_image)

    return image_boxes