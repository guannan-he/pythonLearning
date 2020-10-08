import cv2
import matplotlib.pyplot as plt
import numpy
import math
import subfunctions

# read images andsign a image to process
pic1Name = "20200717190632173"
pic2Name = "20200717190731576"

nameFlag = pic2Name

# image process
imToProcess = cv2.imread(f"src/{nameFlag}.jpg")
imshape = imToProcess.shape  # H, W, channel
gray=subfunctions.grayscale(imToProcess)
kernel_size=5
blur=subfunctions.gaussian_blur(gray, kernel_size)
low_threshold=50
high_threshold=200
edges=subfunctions.canny(blur, low_threshold, high_threshold)
#
vertices=numpy.array([[(0,imshape[0]),(imshape[1]/2-20, imshape[0]/2+50),
                        (imshape[1]/2+20, imshape[0]/2+50),
                    (imshape[1],imshape[0]),(0,500),(960,500)]],
                      dtype=numpy.int32)
partial=subfunctions.region_of_interest(edges,vertices)
#
rho=1
theta=numpy.pi/180
threshold=13
min_line_len=15
max_line_gap=10
lines=subfunctions.hough_lines(partial, rho, theta, threshold, min_line_len, max_line_gap)
#
final=subfunctions.weighted_img(lines,imToProcess)

# show images
imList = [imToProcess, gray, blur, edges, partial, final]
for image in imList:
    subfunctions.show_image_in_window(image)
# save image
cv2.imwrite(f"res/{nameFlag}.jpg", final)
