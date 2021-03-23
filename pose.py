from collections import deque
import numpy as np
import imutils
import cv2
import math
import pandas as pd
import calendar,time

redLower = (161, 177, 176)
redUpper = (181, 197, 256)
blueLower = (110, 50, 50)
blueUpper = (130, 255, 255)

pts = deque(maxlen=0)
counter = 0
(dX, dY) = (0, 0)
direction = ""

coord = []
video_cap = cv2.VideoCapture('asyu_2.mpg')
cv2.namedWindow("Frame")

while (video_cap.isOpened()):
    ret, frame = video_cap.read()
    if ret:
        assert not isinstance(frame, type(None)), 'frame not found'

    if frame is not None:
        frame = imutils.resize(frame, width=720)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, redLower, redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("mask",mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #bu satırda cmax’ı içine alan en küçük çembere ait merkez ve yarıçap bilgilerini buluyoruz.
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            print(center)
            coord.append(dict(x=center[0],
                              y=center[1],
                              time=int(calendar.timegm(time.gmtime())),
                              angle= math.atan2(M["m10"] - M["m00"], M["m01"] - M["m00"]) * 180 / math.pi),
                         )
            print(math.atan2(M["m10"] - M["m00"], M["m01"] - M["m00"]) * 180 / math.pi)
        for i in np.arange(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue

        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(100) & 0xFF
        counter += 1
        if counter == video_cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        if key == ord("q"):
            break

df = pd.DataFrame(coord)
df.to_csv("data.csv",index=False)

video_cap.release()
cv2.destroyAllWindows()
