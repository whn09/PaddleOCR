# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np


def remove_lines(image):

    mask = get_line_removal_mask(image)

    # Calculate (Image || !Mask)
    result_channels = np.zeros(shape=image.shape)
    for ch in range(0, 3):
      result_channels[:,:, ch] = cv.bitwise_or(image[:, :, ch], cv.bitwise_not(mask))
    
    return result_channels


def recognize_tables(fname):

    # Read image
    image = cv.imread(fname, 1)

    mask = get_line_removal_mask(image)
    cv.imwrite(fname[:-4]+"_mask.jpg", mask)

    canny = cv.Canny(mask, 130, 255, 1)
    cv.imwrite(fname[:-4]+"_canny.jpg", canny)

    lines = cv.HoughLines(canny, 1, np.pi/180,118)
    print("lines:", len(lines))
    line_result = image.copy()
    for line in lines:
        rho = line[0][0]  #第一个元素是距离rho
        theta= line[0][1] #第二个元素是角度theta
        # print (rho)
        # print (theta)
        if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
            pt1 = (int(rho/np.cos(theta)),0)               #该直线与第一行的交点
            #该直线与最后一行的焦点
            pt2 = (int((rho-line_result.shape[0]*np.sin(theta))/np.cos(theta)),line_result.shape[0])
            cv.line(line_result, pt1, pt2, (255))             # 绘制一条白线
        else:                                                  #水平直线
            pt1 = (0,int(rho/np.sin(theta)))               # 该直线与第一列的交点
            #该直线与最后一列的交点
            pt2 = (line_result.shape[1], int((rho-line_result.shape[1]*np.cos(theta))/np.sin(theta)))
            cv.line(line_result, pt1, pt2, (255), 1)           # 绘制一条直线
    cv.imwrite(fname[:-4]+"_line.jpg", line_result)

    # Return contours, hierarchy
    cnts, hier = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    result = np.copy(image) #np.zeros(shape=image.shape)
    cells = []
    for i, cnt in enumerate(cnts):
        # Only low-level contours
        if hier[0, i, 2] == -1:
          x,y,w,h = cv.boundingRect(cnt)

          # Constraint for shape
          if w > 10 and h > 10:
            cv.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
            cells.append([(x,y), (x+w,y+h)])

    cv.imwrite(fname[:-4]+"_result.jpg", result)
    return cells


def get_line_removal_mask(image):

    # Convert to grayscale, apply blur, then binarize
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0.7)
    threshed = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                  cv.THRESH_BINARY, 3, 2)
    # cv.imshow('threshed', threshed)
    # cv.waitKey()
    # Invert image
    threshed = ~threshed
    # cv.imshow('~threshed', threshed)
    # cv.waitKey()

    # Morphology
    
    len_long = 35  # default: 35
    len_small = 3  # default: 3
    kernel_h_long = np.ones((1, len_long), np.uint8)
    kernel_v_small = np.ones((len_small, 1), np.uint8)
    kernel_v_long = np.ones((len_long, 1), np.uint8)
    kernel_h_small = np.ones((1, len_small), np.uint8)
    
    # Erode first to get horizontal lines, then dilate them (horizontally and vertically) ...
    # to make them more significant
    result_horiz = cv.erode(threshed, kernel_h_long, iterations=1)
    result_horiz = cv.dilate(result_horiz, kernel_h_long, iterations=3)
    result_horiz = cv.dilate(result_horiz, kernel_v_small, iterations=1)
    # cv.imshow('result_horiz', result_horiz)
    # cv.waitKey()

    # Erode first to get vertical lines, then dilate them (horizontally and vertically) ...
    # to make them more significant
    result_vert = cv.erode(threshed, kernel_v_long, iterations=1)
    result_vert = cv.dilate(result_vert, kernel_v_long, iterations=3)
    result_vert = cv.dilate(result_vert, kernel_h_small, iterations=1)
    # cv.imshow('result_vert', result_vert)
    # cv.waitKey()

    mask = cv.bitwise_or(result_vert, result_horiz)

    # Invert mask back
    mask = ~mask
    return mask


def median(image):
    result = cv.medianBlur(image, 3)
    return result


def bilateral(image):
    result = cv.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
    return result


if __name__ == '__main__':
    # recognize_tables('mmexport1607322541739.pic_hd.jpg')
    recognize_tables('mmexport1607322541739.pic_hd_cropped.jpg')