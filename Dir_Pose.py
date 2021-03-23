import cv2
import numpy as np
import imutils


def FindCircle (img,Lower, Upper, min, max):

    mask = cv2.inRange(hsv, Lower, Upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    res = cv2.bitwise_and(img, img, mask=mask)
    res_1 = cv2.bitwise_and(res, res, mask=mask)
    video_gray = cv2.cvtColor(res_1, cv2.COLOR_BGR2GRAY)
    circle = cv2.HoughCircles(video_gray, cv2.HOUGH_GRADIENT, 1,
                              img.shape[0] / 64,
                              param1=20, param2=10,
                              minRadius=min, maxRadius=max)
    if circle is not None:
        circle = np.uint16(np.around(circle))
        for i in circle[0, :]:
            # Draw outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 255), 2)
            # Draw inner circle edge_grey = np.int0(edge_grey)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 10)

            circle = (i[0],i[1])
            circle=np.int0(circle)
            print("circle coordinate={}".format(circle))
            break
        return circle

frame = cv2.imread("robocup_2.png")
video_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
video_blur = cv2.medianBlur(video_gray, 7)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# definig the range of red color
red_lower = np.array([165, 80, 111], np.uint8)
red_upper = np.array([200, 255, 255], np.uint8)

# finding the range of red,blue and yellow color in the image
red = cv2.inRange(hsv, red_lower, red_upper)


kernal = np.ones((2, 2), "uint8")

red = cv2.dilate(red, kernal)#beyaz yeri filtreledi
cv2.imshow("red red red",red)
res = cv2.bitwise_and(frame, frame, mask=red)
cv2.imshow("red-filtered",res)

ece_guldu=cv2.bitwise_and(res,res,mask=red)
#cv2.imshow("red mask",ece_guldu)
video_gray5 = cv2.cvtColor(ece_guldu, cv2.COLOR_BGR2GRAY)


big_circ = FindCircle(frame,red_lower,red_upper, 8,24)
small_circ = FindCircle(frame,red_lower,red_upper,1,7)

if big_circ is None:
    print("Can not find Big Circle")
    print("Robot Coordinate nearly equal small circle coordinate."
          "Coordinate = {}".format(small_circ))
if small_circ is None:
    print("Can not find Small Circle")
    print("Robot Coordinate nearly equal big circle coordinate."
          "Coordinate = {}".format(big_circ))
else:
    robot_coord = ((big_circ[0] + small_circ[0]) / 2, (big_circ[1] + small_circ[1]) / 2)
    print("robot coordinate = {}".format(robot_coord))
cv2.imshow("img", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()