import cv2
import numpy as np
import datetime
from matplotlib import pyplot as plt


#references: https://docs.opencv.org/4.4.0/
mountPoint = "smb://openmediavault.local/wd_share/"
fileName = ["lulu.jpg", "lena.jpg", "smarties.png"]
pic = cv2.imread(fileName[1])
picShape = pic.shape  # H, W, channel
pass  # capture from camera
# cap = cv2.VideoCapture(0)  # input from IR camera: 2
# # cap.set(3, 1920)
# # cap.set(4, 1080)
# cnt = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # default format is BGR
#     textStr = str(datetime.datetime.now())
#     frame = cv2.putText(frame, textStr, (10, int(cap.get(4) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), cv2.LINE_4)
#     cv2.imshow("na", frame)
#     cnt += 1
#     if cv2.waitKey(25) == ord('q') or cnt >= 1000:
#         cap.release()
#         cv2.destroyAllWindows()
#         break
pass  # draw shapes
# pic = cv2.line(pic, (0,0), (picShape[1], picShape[0]), (255, 0, 0), 2)
# pic = cv2.arrowedLine(pic, (0,picShape[0]), (picShape[1], 0), (0, 0, 255), 2)  # upper left is (0,0)
# pic = cv2.rectangle(pic, (100, 100), (300, 300), (0, 255, 0), 2)  # thickness == -1, fill the area
# pic = cv2.circle(pic, (picShape[1] // 2, picShape[0] // 2), 100, (128, 128, 128), 2)
# pic = cv2.putText(pic, "LuLu", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), cv2.LINE_4)
# emptyPic = np.zeros(picShape, np.uint8)
pass  # mouse event in window
#
#
# def clickEvent(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print(f"{x}, {y}")
#         cv2.putText(pic, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), cv2.LINE_4)
#         cv2.imshow("lulu", pic)
#     elif event == cv2.EVENT_LBUTTONDOWN:
#         blue, green, red = pic[y, x, :]
#         # print(f"{x}, {y}")
#         cv2.putText(pic, f"{red},{green},{blue}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), cv2.LINE_4)
#         cv2.imshow("lulu", pic)
#     return
#
#
# cv2.imshow("lulu", pic)
# cv2.setMouseCallback("lulu", clickEvent)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # math operation
# picLena = cv2.imread("lena.jpg")
# picLena = cv2.resize(picLena, (picShape[1], picShape[0]))
# # pic = cv2.add(pic, picLena)
# # pic = cv2.bitwise_or(pic, picLena)
# # pic = cv2.bitwise_and(pic, picLena)
# # pic = cv2.bitwise_xor(pic, picLena)
# pic = cv2.bitwise_not(pic)  # value = 255 - value
#
pass  # HSV color space
#
#
# def nothing(x):
#     pass
#
# cv2.namedWindow("track")
# cv2.createTrackbar("LH", "track", 0, 255, nothing)
# cv2.createTrackbar("LS", "track", 0, 255, nothing)
# cv2.createTrackbar("LV", "track", 0, 255, nothing)
# cv2.createTrackbar("UH", "track", 255, 255, nothing)
# cv2.createTrackbar("US", "track", 255, 255, nothing)
# cv2.createTrackbar("UV", "track", 255, 255, nothing)
# cap = cv2.VideoCapture(0)
# # hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
# while True:
#     status, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lh = cv2.getTrackbarPos("LH", "track")
#     ls = cv2.getTrackbarPos("LS", "track")
#     lv = cv2.getTrackbarPos("LV", "track")
#     uh = cv2.getTrackbarPos("UH", "track")
#     us = cv2.getTrackbarPos("US", "track")
#     uv = cv2.getTrackbarPos("UV", "track")
#     l_b = np.array([lh, ls, lv])
#     u_b = np.array([uh, us, uv])
#     mask = cv2.inRange(hsv, l_b, u_b)
#     res = cv2.bitwise_and(frame, frame, mask = mask)
#     cv2.imshow("na", res)
#     if cv2.waitKey(10) == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()
#
pass  # binarization method
# gradient = cv2.imread("gradient.png")
# # _, th1 = cv2.threshold(gradient, 45, 255, cv2.THRESH_BINARY)  # greater than val1, set as val2
# # _, th1 = cv2.threshold(gradient, 45, 255, cv2.THRESH_BINARY_INV)  # less than val1, set as val2
# # _, th1 = cv2.threshold(gradient, 128, 0, cv2.THRESH_TRUNC)  # set pixel val > val1 as val1, val2 invalid
# # _, th1 = cv2.threshold(gradient, 128, 255, cv2.THRESH_TOZERO)  # set pixel val < val1 as 0, val2 invalid
# # _, th1 = cv2.threshold(gradient, 128, 255, cv2.THRESH_TOZERO_INV)  # set pixel val > val1 as 0, val2 invalid
# sudoku = cv2.imread("sudoku.png",0)  # must be binary
# # th1 = cv2.adaptiveThreshold(sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)  #
# th1 = cv2.adaptiveThreshold(sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)  #
#
# cv2.imshow("sudoku", sudoku)
# cv2.imshow("threshold", th1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # 数学形态学（Mathematical morphology）膨胀, 腐蚀
# org = cv2.imread("smarties.png", 0)
# _, mask = cv2.threshold(org, 220, 255, cv2.THRESH_BINARY_INV)
# kernelSize = 2
# iterationNum = 5
# kernel = np.ones((kernelSize, kernelSize), np.uint8)
# dilation = cv2.dilate(mask, kernel, iterations=iterationNum)
# erosion = cv2.erode(mask, kernel, iterations=iterationNum)
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterationNum)  # erosion + dilation
# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterationNum)  # dilation + erosion
# morphGradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel, iterations=iterationNum)  # difference between dilation and erosion
# topHat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel, iterations=iterationNum)  # difference of source and opening
#
# titles = ["org", "mask", "dilation", "erosion", "opening", "closing", "morphGradient", "topHat"]
# images = [org, mask, dilation, erosion, opening, closing, morphGradient, topHat]
#
# for i in range(len(titles)):
#     plt.subplot(2, 4, i + 1)
#     plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
pass  # blurring amd smoothing
# pic = cv2.cvtColor(cv2.imread("lena.jpg"), cv2.COLOR_BGR2RGB)
# kernelSize = 5
# kernel = np.ones((kernelSize, kernelSize), np.float32) / (kernelSize ** 2)
# homo = cv2.filter2D(pic, -1, kernel)  # arg 2: depth
# blur = cv2.blur(pic, (kernelSize, kernelSize))
# gaussian = cv2.GaussianBlur(pic, (5, 5), 0)
# median = cv2.medianBlur(pic, 5)  # pepper and salt use median
# bilateral = cv2.bilateralFilter(pic, 9, 75, 75)
#
# titles = ["pic", "homo", "blur", "gaussian", "median", "bilateral"]
# images = [pic, homo, blur, gaussian, median, bilateral]
#
# for i in range(len(titles)):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(images[i])
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
pass  # gradiant edge detection
# pic = cv2.imread("messi5.jpg", 0)
# sobelX = cv2.Sobel(pic, cv2.CV_64F, 1, 0, ksize=3)  # val1: x derivative order, val2: x derivative order
# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = cv2.Sobel(pic, cv2.CV_64F, 0, 1, ksize=3)
# sobelY = np.uint8(np.absolute(sobelY))
# lap = cv2.Laplacian(pic, cv2.CV_64F, ksize=3)
# lap = np.uint8(np.absolute(lap))
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
# canny = cv2.Canny(pic, 100, 200)
#
# titles = ["pic", "sobelX", "sobelY", "lap", "sobelCombined", "canny"]
# images = [pic, sobelX, sobelY, lap, sobelCombined, canny]
#
# for i in range(len(titles)):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
pass  # canny edge detection
#
#
# def nothing(x):
#     pass
#
# # pic = cv2.cvtColor(cv2.imread("lulu.jpg"), cv2.COLOR_BGR2RGB)
#
#
# pic = cv2.imread("lulu.jpg")
# cv2.namedWindow("track")
# cv2.createTrackbar("lwr", "track", 0, 255, nothing)
# cv2.createTrackbar("upr", "track", 255, 255, nothing)
# cv2.imshow("org", pic)
# while True:
#     lwr = cv2.getTrackbarPos("lwr", "track")
#     upr = cv2.getTrackbarPos("upr", "track")
#     canny = cv2.Canny(pic, lwr, upr)
#     cv2.imshow("canny", canny)
#     if cv2.waitKey(10) == ord("q"):
#         break
# cv2.destroyAllWindows()
#
pass  # subsampling, image pyramid
# lr = cv2.pyrDown(pic)
# up = cv2.pyrUp(lr)
# lap = cv2.subtract(pic, up)  # laplacian pyramid result
# lap_visual = cv2.cvtColor(lap, cv2.COLOR_BGR2GRAY)
# lap_visual = cv2.threshold(lap_visual, 1, 255, cv2.THRESH_BINARY)
# lap_visual = cv2.cvtColor(lap_visual[1], cv2.COLOR_GRAY2BGR)
# img = [pic, lr, up, lap, lap_visual]
# title = ["pic", "lr(gaussian)", "up(gaussian)", "lap", "lap_visual"]
# for i in range(len(img)):
#     tmp = img[i]
#     cv2.putText(tmp, title[i], (0, tmp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), cv2.LINE_4)
#     cv2.imshow("result", tmp)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # image blending using image pyramid
# apple = cv2.imread("apple.jpg")
# orange = cv2.imread("orange.jpg")
# apple_orange = np.hstack((apple[:, 0:256], orange[:, 256:]))
#
# # gaussian pyramid down for apple
# apple_copy = apple.copy()
# apple_group = [apple_copy]
# for i in range(6):
#     apple_copy = cv2.pyrDown(apple_copy)
#     apple_group.append(apple_copy)
# # gaussian pyramid down for orange
# orange_copy = orange.copy()
# orange_group = [orange_copy]
# for i in range(6):
#     orange_copy = cv2.pyrDown(orange_copy)
#     orange_group.append(orange_copy)
#
# # laplacian for apple
# apple_copy = apple_group[5]
# apple_lp = [apple_copy]
# for i in range(5, 0, -1):
#     expand = cv2.pyrUp(apple_group[i])
#     lap = cv2.subtract(apple_group[i - 1], expand)
#     apple_lp.append(lap)
#
# # laplacian for orange
# orange_copy = orange_group[5]
# orange_lp = [orange_copy]
# for i in range(5, 0, -1):
#     expand = cv2.pyrUp(orange_group[i])
#     lap = cv2.subtract(orange_group[i - 1], expand)
#     orange_lp.append(lap)
#
# # laplacian for apple_orange
# apple_orange_prymaid = []
# n = 0
# for apple_lap, orange_lap in zip(apple_lp, orange_lp):
#     n += 1
#     col, rol, ch = apple_lap.shape
#     laplacian = np.hstack((apple_lap[:,0:col // 2], orange_lap[:, col // 2:]))
#     apple_orange_prymaid.append(laplacian)
#
# # reconstruct
# apple_orange_recostruct = apple_orange_prymaid[0]
# for i in range(1, 6):
#     apple_orange_recostruct = cv2.pyrUp(apple_orange_recostruct)
#     apple_orange_recostruct = cv2.add(apple_orange_prymaid[i], apple_orange_recostruct)
#
#
# img = [apple, orange, apple_orange, apple_orange_recostruct]
# title = ["apple", "orange", "apple_orange", "apple_orange_recostruct"]
# plt.figure("result")
# for i in range(len(img)):
#     plt.subplot(2, 3, i + 1)
#     tmp = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
#     plt.imshow(tmp)
#     plt.title(title[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
# plt.figure("laplacian pyramid")
# for i in range(len(apple_orange_prymaid)):
#     # pass
#     plt.subplot(1,6,i + 1)
#     tmp = apple_orange_prymaid[i]
#     # tmp = cv2.bitwise_or(tmp, np.zeros(tmp.shape, np.uint8), 20)
#     tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
#
#     plt.imshow(tmp)
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
pass  # image contours
# img = cv2.imread("opencv-logo.png")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # using binary image find threshold
# ret, thresh = cv2.cv2.threshold(imgGray, 63, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(f"contour number: {str(len(contours))}")
# res = cv2.drawContours(img.copy(), contours, -1, (255, 255, 255), 5)
#
# imag = [img, imgGray, thresh, res]
# title = ["cvLogo", "imgGray", "thresh", f"{str(len(contours))} contours founded"]
# plt.figure("results")
# for i in range(len(imag)):
#     plt.subplot(2, 3, i + 1)
#     tmp = cv2.cvtColor(imag[i], cv2.COLOR_BGR2RGB)
#     plt.imshow(tmp)
#     plt.title(title[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
pass  # motion detection
# cap = cv2.VideoCapture("vtest.avi")
# _, frame1 = cap.read()
# _, frame2 = cap.read()
#
# while cap.isOpened():
#     # read and preprocess
#     diff = cv2.absdiff(frame1, frame2)
#     frame1 = frame2
#     _, frame2 = cap.read()
#     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     # motion detection
#     kernelSize = 11
#     blur = cv2.GaussianBlur(gray, (kernelSize, kernelSize), 0)
#     _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((kernelSize, kernelSize), np.uint8)
#     dilated = cv2.dilate(thresh, None, iterations=3)
#     contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     rect = frame1.copy()
#     flag = False
#     for contour in contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         if w * h < 900:
#             continue
#         rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         flag = True
#     if flag:
#         rect = cv2.putText(rect, "have moving object", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), cv2.LINE_4)
#     else:
#         rect = cv2.putText(rect, "no moving object", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), cv2.LINE_4)
#     draw_contour = cv2.drawContours(frame1.copy(), contours, -1, (0, 255, 0), 2)
#     # show frame
#     cv2.imshow("original", frame1)
#     cv2.imshow("diff", dilated)
#     cv2.imshow("contour", draw_contour)
#     cv2.imshow("rectangle", rect)
#     if cv2.waitKey(40) == ord("q"):
#         print("interrupted by user")
#         break
# cv2.destroyAllWindows()
# cap.release()
#
pass  # shape detection
# shape = cv2.imread("shapes.jpg")
# shapesGray = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(shapesGray, 63, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# thresCopy = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
#     thresCopy = cv2.drawContours(thresCopy, [approx], -1, (0, 0, 255), 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]
#     if len(approx) == 3:
#         str = "triangle"
#     elif len(approx) == 4:
#         _, _, w, h = cv2.boundingRect(approx)
#         if abs(w / h - 1) < 0.1:
#             str = "square"
#         else:
#             str = "rectangle"
#     elif len(approx) == 5:
#         str = "pentagon"
#     elif len(approx) == 10:
#         str = "star"
#     else:
#         str = "circle"
#     thresCopy = cv2.putText(thresCopy, str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), cv2.LINE_4)
#
#
# cv2.imshow("shapes", shape)
# cv2.imshow("binary", thresh)
# cv2.imshow("contours", thresCopy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # histogram
# # img = np.zeros((200, 200), np.uint8)
# # img = cv2.imread("lulu.jpg")
# img = cv2.imread("lena.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist([img], [0], None, [256], [0, 255])
# plt.plot(hist)
#
# # plt.hist(img.ravel(), 256, [0, 255])
# plt.show()
#
pass  # template matching

#
pass  #
#
pass  #
#
pass  #
#
if False:
    cv2.imshow("na", pic)
    key = cv2.waitKey(0)
    if key == ord('s'):
        cv2.imwrite("result.jpg", pic)
    cv2.destroyAllWindows()
exit(0)