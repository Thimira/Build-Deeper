import numpy as np
import cv2
import dlib

video_capture = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()

img = cv2.imread('..//images//Face.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rects = detector(img_gray, 0)

for rect in rects:
    x = rect.left()
    y = rect.top()
    x1 = rect.right()
    y1 = rect.bottom()

    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

# Display the resulting image
cv2.imshow('Detected Faces', img)

# Wait for a keypress
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

