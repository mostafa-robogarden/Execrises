import cv2
import numpy as np
import matplotlib.pyplot as plt

def displayImg(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
def main():
    image = cv2.imread('cute.jpg')
    image = cv2.resize(src = image, dsize=(600, 800), interpolation=cv2.INTER_CUBIC)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    displayImg(image)
main()