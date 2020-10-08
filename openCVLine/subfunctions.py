import cv2
import numpy


def show_image_in_window(image):
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input image", image)
    cv2.waitKey(0)


def grayscale(img):
    """
    将图像处理为灰度图像，因为使用cv2read所以要用BGR进行转换
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):  # 返回image，边缘部分为255，其余为0

    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = numpy.zeros_like(img)
    # print("mask_shape", mask.shape)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # show_image_in_window(mask)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines):
    color = [0, 255, 0]  # RGB
    thickness = 5
    left_lines_x = []
    left_lines_y = []
    right_lines_x = []
    right_lines_y = []
    line_y_max = 0
    line_y_min = 999
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 > line_y_max:
                line_y_max = y1
            if y2 > line_y_max:
                line_y_max = y2
            if y1 < line_y_min:
                line_y_min = y1
            if y2 < line_y_min:
                line_y_min = y2
            k = (y2 - y1) / (x2 - x1)
            if k < -0.3:
                left_lines_x.append(x1)
                left_lines_y.append(y1)
                left_lines_x.append(x2)
                left_lines_y.append(y2)
            elif k > 0.3:
                right_lines_x.append(x1)
                right_lines_y.append(y1)
                right_lines_x.append(x2)
                right_lines_y.append(y2)
    # 最小二乘直线拟合
    left_line_k, left_line_b = numpy.polyfit(left_lines_x, left_lines_y, 1)
    right_line_k, right_line_b = numpy.polyfit(right_lines_x, right_lines_y, 1)
    # 根据直线方程和最大、最小的y值反算对应的x
    cv2.line(img, (int((line_y_max - left_line_b) / left_line_k), line_y_max),
             (int((line_y_min - left_line_b) / left_line_k), line_y_min), color, thickness)
    cv2.line(img, (int((line_y_max - right_line_b) / right_line_k), line_y_max),
             (int((line_y_min - right_line_b) / right_line_k), line_y_min), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, numpy.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # print(lines.shape)
    line_img = numpy.zeros((img.shape[0], img.shape[1], 3), dtype=numpy.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, a=0.8, b=1., v=0.):
    return cv2.addWeighted(initial_img, a, img, b, v)