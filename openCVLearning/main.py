import cv2
import numpy as np
import datetime
#references: https://docs.opencv.org/4.4.0/
mountPoint = "smb://openmediavault.local/wd_share/"
fileName = ["lulu.jpg", "lena.jpg", "smarties.png"]
pic = cv2.imread(fileName[2])
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
def nothing(x):
    pass
cv2.namedWindow("track")
cv2.createTrackbar("LH", "track", 0, 255, nothing)
cv2.createTrackbar("LS", "track", 0, 255, nothing)
cv2.createTrackbar("LV", "track", 0, 255, nothing)
cv2.createTrackbar("UH", "track", 255, 255, nothing)
cv2.createTrackbar("US", "track", 255, 255, nothing)
cv2.createTrackbar("UV", "track", 255, 255, nothing)
cap = cv2.VideoCapture(0)
# hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
while True:
    status, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lh = cv2.getTrackbarPos("LH", "track")
    ls = cv2.getTrackbarPos("LS", "track")
    lv = cv2.getTrackbarPos("LV", "track")
    uh = cv2.getTrackbarPos("UH", "track")
    us = cv2.getTrackbarPos("US", "track")
    uv = cv2.getTrackbarPos("UV", "track")
    l_b = np.array([lh, ls, lv])
    u_b = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow("na", res)
    if cv2.waitKey(10) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
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