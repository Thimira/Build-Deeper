import numpy as np
import cv2
import dlib

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if (ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()

            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

    ch = 0xFF & cv2.waitKey(1)

    # press "q" to quit the program.
    if ch == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
