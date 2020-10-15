import cv2
import numpy as np
import datetime
from matplotlib import pyplot as plt


# references: https://docs.opencv.org/4.4.0/
#             https://www.youtube.com/playlist?list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K
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
# get messi and his face
# messi = cv2.imread("messi5.jpg")
#
#
# def printLocation(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"{x}, {y}")
#
# messiFace = np.stack((messi[92:131, 223:258]))
# grayMessi = cv2.cvtColor(messi, cv2.COLOR_BGR2GRAY)
# grayFace = cv2.cvtColor(messiFace, cv2.COLOR_BGR2GRAY)
# res = cv2.matchTemplate(grayMessi, grayFace, cv2.TM_CCORR_NORMED)
# threshold = 0.98
# loc = np.where(res >= threshold)
# print(f"{len(loc[0])} results matched" )
# for i in range(len(loc[0])):
#     x = loc[0][i]
#     y = loc[1][i]
#     messi = cv2.rectangle(messi, (y, x), (y + grayFace.shape[1], x + grayFace.shape[0]), (0, 255, 0), 3)
# cv2.imshow("pic", messi)
# cv2.imshow("messi face", messiFace)
# cv2.setMouseCallback("pic", printLocation)
# # find image
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # hgn face matching
# # get a hgn face
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# cap = cv2.VideoCapture(0)
# _, frame = cap.read()
# faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
# print(f"{len(faces)} faces were detected")
# x, y, w, h = 0, 0, 0, 0
# # maximum face area
# for face in faces:
#     if face[2] * face[3] > w * h:
#         x, y, w, h = face
# hgnFace = frame[y: y + h, x: x + w]
# grayFace = cv2.cvtColor(hgnFace, cv2.COLOR_BGR2GRAY)
# cv2.imshow("hgnFace", hgnFace)
# # cv2.waitKey(0)
# threshold = 0.8
# while cap.isOpened():
#     _, frame = cap.read()
#     grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rx, ry, rw, rh = 0, 0, 0, 0
#     ROIs = face_cascade.detectMultiScale(grayFrame, 1.1, 4)
#     # print(f"{len(ROIs)} ROIs were detected")
#     for ROI in ROIs:
#         if ROI[2] * ROI[3] > rw * rh:
#             rx, ry, rw, rh = ROI
#     tmpGrayFace = cv2.resize(grayFace, (rw, rh))
#     w = tmpGrayFace.shape[1]
#     h = tmpGrayFace.shape[0]
#     # expand roi
#     rx -= 20; ry -= 20; rw += 40; rh += 40
#
#     finalROI = grayFrame[ry: ry + rh, rx: rx + rw]
#     finalROI_color = frame[ry: ry + rh, rx: rx + rw]
#     res = cv2.matchTemplate(finalROI, tmpGrayFace, cv2.TM_CCORR_NORMED)
#     loc = np.where(res >= threshold)
#     print(f"{len(loc[0])} results matched")
#     # draw faces on color roi
#     if len(loc[0]) > 0:
#         x = loc[0][0]
#         y = loc[1][0]
#         cv2.rectangle(finalROI_color, (y, x), (y + w, x + h), (0, 255, 0), 2)
#     frame[ry: ry + rh, rx: rx + rw] = finalROI_color
#     cv2.imshow("capture", frame)
#     if cv2.waitKey(40) == ord("q"):
#         break
# cv2.destroyAllWindows()
#
pass  # hough transform (hgn)
# blank = np.zeros((512, 512), np.uint8)
# lineImage = cv2.line(blank.copy(), (0, 0), (511, 511), 255)
# box = np.zeros((1536, 1536), np.uint8)
# for i in range(512):
#     for j in range(512):
#         if lineImage[i][j] == 0:
#             continue
#         else:
#             for k in range(1536):
#                 theta = k / 1536 * np.pi
#                 val = i * np.cos(theta) + j * np.sin(theta)
#                 box[k][int(val) + 768] += 1
# print(str(box.max()))
# loc = np.where(box == box.max())
# theta = loc[0][0] * np.pi / 1536
# rou = loc[1][1] - 768
# a = -1 / np.tan(theta)
# b = rou / np.sin(theta)
# pt1 = (0, np.uint(0 * a + b))
# pt2 = (511, np.uint(511 * a + b))
# afterHough = cv2.line(blank.copy(), pt1, pt2, (255))
# print(f"theta = {theta}, rou = {rou}")
# _, box = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY)
# cv2.imshow("org", lineImage)
# cv2.imshow("hough", box)
# cv2.imshow("afterHough", afterHough)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # hough transform (indian tutorial)
# filePath = ["../openCVLine/src/20200717190731576.jpg", "../openCVLine/src/20200717190632173.jpg","sudoku.png"]
# pic = cv2.imread(filePath[1])
# canny = cv2.Canny(cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY), 50, 150, apertureSize=3)
# lines = cv2.HoughLines(canny, 1, np.pi / 180, 150)  # returns rou and theta, val1, val2 are resolution, val3, is thresh
# addLine = pic.copy()
# for line in lines:
#     rou, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rou
#     y0 = b * rou
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 + 1000 * (b))
#     y2 = int(y0 + 1000 * (-a))
#     cv2.line(addLine, (x1, y1), (x2, y2), (255, 0, 0), 2)
# addLineP = pic.copy()
# linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)  # returns start point and end point
# for lineP in linesP:
#     x1, y1, x2, y2 = lineP[0]
#     cv2.line(addLineP, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# cv2.imshow("pic", pic)
# cv2.imshow("canny", canny)
# cv2.imshow("addLine", addLine)
# cv2.imshow("addLineP", addLineP)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # lane detection
# filePath = ["../openCVLine/src/20200717190731576.jpg", "../openCVLine/src/20200717190632173.jpg","sudoku.png"]
# pic = cv2.imread(filePath[0])
# gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
# h, w = gray.shape
# ROI_Vertices = np.array([[0, h],
#                           [0, h - 20],
#                           [w // 2 - 20, h // 2 - 20],
#                           [w // 2 + 20, h // 2 - 20],
#                           [w, h - 20],
#                           [w, h]])
# mask = cv2.fillPoly(np.zeros([h, w], np.uint8), [ROI_Vertices], 255)
# gray = cv2.blur(gray, (9, 9))
# _, gray = cv2.threshold(gray, None, 255, cv2.THRESH_OTSU)
# canny = cv2.Canny(gray, 150, 250)
# canny = cv2.bitwise_and(canny, mask)
# linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 180, minLineLength=50, maxLineGap=15)
# for line in linesP:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(pic, (x1, y1), (x2, y2), (0, 255, 0), 5)
# cv2.imshow("pic", pic)
# cv2.imshow("gray", gray)
# cv2.imshow("canny", canny)
#
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # lane detection on video frame
#
#
# def markLine(img, vertices):
#     blur = cv2.blur(img, (9, 9))
#     gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, None, 255, cv2.THRESH_OTSU)
#     roi = cv2.bitwise_and(thresh, vertices)
#     canny = cv2.Canny(roi, 100, 100)
#     lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=30)
#     # print(len(lines))
#     # use 3rd polyline
#     left_x = []
#     left_y = []
#     right_x = []
#     right_y = []
#     # assume x2 always >= x1
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0))
#         # print(f"{x1},{x2}:{y1},{y2}:{x1 >= x2},{y1 >= y2}")
#         if 2 * abs(y2 - y1) - (x2 - x1) < 0:
#             continue
#         elif y1 > y2:
#             left_x.append(x1)
#             left_x.append(x2)
#             left_y.append(y1)
#             left_y.append(y2)
#         else:
#             right_x.append(x1)
#             right_x.append(x2)
#             right_y.append(y1)
#             right_y.append(y2)
#     if len(left_x) > 3:
#         paraL = np.polyfit(left_x, left_y, 3)
#         left_x = np.sort(left_x)
#         left_y = ((paraL[0] * left_x + paraL[1]) * left_x + paraL[2]) * left_x + paraL[3]
#         pt1 = (left_x[0], int(left_y[0]))
#         for i in range(len(left_x) - 1):
#             pt2 = (left_x[i + 1], int(left_y[i + 1]))
#             cv2.line(img, pt1, pt2, (0, 255, 0), 5)
#             pt1 = pt2
#     if len(right_x) > 3:
#         paraR = np.polyfit(right_x, right_y, 3)
#         right_x = np.sort(right_x)
#         right_y = ((paraR[0] * right_x + paraR[1]) * right_x + paraR[2]) * right_x + paraR[3]
#         pt1 = (right_x[0], int(right_y[0]))
#         for i in range(len(right_x) - 1):
#             pt2 = (right_x[i + 1], int(right_y[i + 1]))
#             cv2.line(img, pt1, pt2, (0, 255, 0), 5)
#             pt1 = pt2
#     return img
#
#
# fileName = ["straight.mp4", "curvature.mp4"]
# cap = cv2.VideoCapture(fileName[1])
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# roiVertices = np.array([[20, h],
#                         [w // 2 - 20, int(h * 0.6)],
#                         [w // 2 + 20, int(h * 0.6)],
#                         [w - 20, h]], dtype=np.int32)
# roiMask = cv2.fillPoly(np.zeros([h, w], np.uint8), [roiVertices], 255)
# if False:
#     _, frame = cap.read()
#     cap.release()
#     cv2.imshow("frame", roiMask)
#     processedFrame = markLine(frame, roiMask)
#     cv2.imshow("processed", processedFrame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     exit(0)
# while cap.isOpened():
#     isRead, frame = cap.read()
#     if not isRead:
#         print("exited by EOF")
#         break
#     cv2.imshow("frame", frame)
#     processedFrame = markLine(frame, roiMask)
#     cv2.imshow("processed", processedFrame)
#     if cv2.waitKey(10) == ord("q"):
#         print("exited by user")
#         break
# cap.release()
# cv2.destroyAllWindows()
#
pass  # hough circle transform
# smarties = cv2.imread("smarties.png")
# gray = cv2.cvtColor(smarties, cv2.COLOR_BGR2GRAY)
# gray = cv2.medianBlur(gray, 5)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0, maxRadius=0)
# detected_circles = np.uint16(np.around(circles))
# for x, y, r in detected_circles[0,:]:
#     cv2.circle(smarties, (x, y), r, (255, 0, 255), 5)
#
# cv2.imshow("smarties", smarties)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  # haar cascade classifier
# # haar comes with a trainer and a classifier
# # references https://github.com/opencv/opencv/tree/master/data/haarcascades
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.2, 4)
#     for face in faces:
#         x, y, w, h = face
#         # to make sure no eyes outside the face
#         roi_gray = gray[y: y + h, x: x + w]
#         roi_color = frame[y: y + h, x: x + w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
#         for eye in eyes:
#             x, y, w, h = eye
#             cv2.rectangle(roi_color, (x, y), (x + w, y + h), (255, 0, 0), 5)
#         # replace back eye detection result
#         x, y, w, h = face
#         frame[y: y + h, x: x + w] = roi_color
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
#         cv2.imshow("face", frame)
#
#     if cv2.waitKey(20) == ord("q"):
#         print("exit by user")
#         break
# cap.release()
# cv2.destroyAllWindows()
#
pass  # harris corner detection & Shi Tomasi corner detection
# chess = cv2.imread("chessboard.png")
# gray = cv2.cvtColor(chess, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# dst = cv2.dilate(dst, None)
# chessHarris = chess.copy()
# chessHarris[dst > 0.01 * dst.max()] = [0, 0, 255]
# corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
# chessShi = chess.copy()
# for corner in corners:
#     x, y = np.int32(corner.ravel())
#     cv2.circle(chessShi, (x, y), 5, (0, 255, 0), 5)
# cv2.imshow("chessHarris", chessHarris)
# cv2.imshow("Shi Tomasi", chessShi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
pass  #
#
pass  #
#
pass  #
#
pass  #
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