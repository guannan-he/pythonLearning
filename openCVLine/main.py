# references: https://blog.csdn.net/weixin_40693859/article/details/95009497
#             https://blog.csdn.net/weixin_40693859/article/details/95011875

import cv2
import numpy
import subfunctions

# read images and assign a image to process
pic1Name = "20200717190632173"
pic2Name = "20200717190731576"

nameFlag = pic1Name

# image process
imToProcess = cv2.imread(f"src/{nameFlag}.jpg")
imshape = imToProcess.shape  # H, W, channel
gray = subfunctions.grayscale(imToProcess)
kernel_size = 9
blur = subfunctions.gaussian_blur(gray, kernel_size)
#
low_threshold = 50
high_threshold = 200
edges = subfunctions.canny(blur, low_threshold, high_threshold)
#
vertices = numpy.array([[(0, imshape[0]), (imshape[1] / 2 - 20, imshape[0] / 2 + 50),
                         (imshape[1] / 2 + 20, imshape[0] / 2 + 50),
                         (imshape[1], imshape[0])]],
                       dtype=numpy.int32)  # rectangle area
partial = subfunctions.region_of_interest(edges, vertices)
#
rho = 1
theta = numpy.pi / 180
threshold = 13
min_line_len = 15
max_line_gap = 10
lines = subfunctions.hough_lines(partial, rho, theta, threshold, min_line_len, max_line_gap)
#
final = subfunctions.weighted_img(lines, imToProcess)

# show images
if True:
    imList = [imToProcess, gray, blur, edges, partial, lines, final]
    for image in imList:
        subfunctions.show_image_in_window(image)
    cv2.destroyAllWindows()
# save image
cv2.imwrite(f"res/{nameFlag}.jpg", final)
