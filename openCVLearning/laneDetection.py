import cv2
import numpy as np
# 1280 * 720 only
calibrateCamera = 1
singleFrameMode = True
isAccelerate = True
fileNames = ["harder_challenge_video.mp4", "challenge_video.mp4", "curvature.mp4"]
camPath = "./camera_cal/calibration"
videoFile = fileNames[2]
camCaliRate = 0.5
canny_l = 100
canny_h = 200
warpSize = (500, 600)


def calibrateCameraFunc(size1, size2, w, h, thresh, framePath, frameCnt, format = ".jpg"):
    camCnt = range(1, frameCnt + 1)
    objPoints = []  # point in real wold
    imgPoints = []  # points in image
    objP = np.zeros((size1 * size2, 3), np.float32)
    objP[:, :2] = np.mgrid[0:size1, 0:size2].T.reshape(-1,2)
    foundCnt = 0
    for i in camCnt:
        frameName = f"{framePath}{i}{format}"
        # print(frameName)
        frame = cv2.imread(frameName)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(frameGray, (size1, size2))
        if ret:
            foundCnt += 1
            objPoints.append(objP)
            imgPoints.append(corners)
            cv2.drawChessboardCorners(frame, (size1, size2), corners, ret)
            cv2.imshow("chess pattern", frame)
            cv2.waitKey(10)
    print(f"{foundCnt} frames were used to calibrate camera")
    cv2.destroyAllWindows()
    ret, mtx, dist, rVecs, tVecs = cv2.calibrateCamera(objPoints, imgPoints, frameGray.shape[::-1], None, None)
    # camMat, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), thresh, (w, h))
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, camMat, (w, h), 5)
    # return (mapx, mapy, roi)
    return (mtx, dist)


def find_pt_inline(p1, p2, y):
    m_inv = (p2[0] - p1[0]) / float(p2[1] - p1[1])
    dy = (y - p1[1])
    x = p1[0] + m_inv * dy
    return [x, y]


def getPerspectPara(canny_l, canny_h, w, h, ifCali, mtx, dist, warpSize):
    img = cv2.imread("./test_images/straight_lines1.jpg")
    if ifCali:
        img = cv2.undistort(img, mtx, dist, None, mtx)
    maskPoly = np.array([[0, h - 50],
                        [w, h - 50],
                        [w / 2, h / 2 + 50]],np.int32)
    mask = cv2.fillPoly(np.zeros((h, w), np.uint8), [maskPoly],255)
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    canny = cv2.Canny(imgHLS[:, :, 1], canny_h, canny_l)
    canny = cv2.bitwise_and(canny, mask)
    lines = cv2.HoughLinesP(canny, 0.5, np.pi / 180, 20, minLineLength=180, maxLineGap=120)
    Lhs = np.zeros((2, 2), np.float32)
    Rhs = np.zeros((2, 1), np.float32)
    x_max = 0
    x_min = 2555
    for line in lines:
        x1, y1, x2, y2 = line[0]
        normal = np.array([[y1 - y2],[x2 - x1]], np.float32)
        unitNorm = normal / np.linalg.norm(normal)
        pt = np.array([[x1], [y1]], np.float32)
        outer = np.matmul(unitNorm, unitNorm.T)
        Lhs += outer
        Rhs += np.matmul(outer, pt)
        x_iter_max = max(x1, x2)
        x_iter_min = min(x1, x2)
        x_max = max(x_max, x_iter_max)
        x_min = min(x_min, x_iter_min)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
    vp = np.matmul(np.linalg.inv(Lhs), Rhs)
    width = x_max - x_min
    top = vp[1] + 65
    bot = frameH - 40
    width = 500
    p1 = [vp[0] - width / 2, top]
    p2 = [vp[0] + width / 2, top]
    p3 = find_pt_inline(p2, vp, bot)
    p4 = find_pt_inline(p1, vp, bot)
    src_pts = np.float32([p1, p2, p3, p4])
    dst_pts = np.float32([[0, 0], [warpSize[0], 0],
                          [warpSize[0], warpSize[1]],
                          [0, warpSize[1]]])
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, warpSize)
    if 0:
        print('vp is : ', vp)
        cv2.circle(img, (int(vp[0]), int(vp[1])), 5, (255, 255, 255), -1)
        cv2.imshow("mask", img)
        cv2.imshow("warped", warped)
        cv2.waitKey(0)
    return (M, M_inv)


cap = cv2.VideoCapture(videoFile)
frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


if calibrateCamera:
    if isAccelerate:
        mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02], [0.00000000e+00, 1.15282291e+03, 3.86128937e+02],
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist = np.array([[-0.24688775, -0.02373134, -0.00109842, 0.00035108, -0.00258569]])
    else:
        mtx, dist = calibrateCameraFunc(6, 9, frameW, frameH, camCaliRate, camPath, 20)
else:
    mtx = dist = None
if isAccelerate:
    M = np.array([[-1.54102761e-01, -5.92702913e-01, 3.48573753e+02], [-1.39257879e-16, -1.90105835e+00, 9.25428566e+02],
         [-1.08436207e-19, -2.37081154e-03, 1.00000000e+00]])
    M_inv = np.array([[1.00000000e+00, -7.97723538e-01, 3.89662354e+02], [0.00000000e+00, -5.26022815e-01, 4.86796509e+02],
             [0.00000000e+00, -1.24710093e-03, 1.00000000e+00]])
else:
    M, M_inv = getPerspectPara(canny_l, canny_h, frameW, frameH, calibrateCamera, mtx, dist, warpSize)
##############################################################################################

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("EOF")
        break
    if calibrateCamera:
        remap = cv2.undistort(frame, mtx, dist, None, mtx)
    else:
        remap = frame.copy()

    wraped = cv2.warpPerspective(frame, M, warpSize)

    cv2.imshow("org", frame)
    cv2.imshow("remap", remap)
    cv2.imshow("wraped", wraped)
    key = cv2.waitKey(20)
    if key == ord("q"):
        print("user exit")
        break
    elif key == ord(" "):
        cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
exit(0)