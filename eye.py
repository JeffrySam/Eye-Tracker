import cv2
import dlib
import math


# function to return midpoint between two coordinate points
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


# function to return distance between two coordinate points
def distance(x1, x2, y1, y2):
    delta_x = (x2-x1)**2
    delta_y = (y2-y1)**2
    return math.sqrt(delta_x+delta_y)


cap = cv2.VideoCapture(0)

# Using DLib library's facial landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    cv2.rectangle(frame, (450, 150), (250, 100), (255, 0, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "keep eyes within the blue rectangle", (50, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    for face in faces:
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        landmarks = predictor(gray, face)

        # closeup_one and closeup_two used to get a frame with a closer view of the eye
        closeup_one = (landmarks.part(17).x, landmarks.part(17).y)
        closeup_two = (landmarks.part(26).x, landmarks.part(15).y)
        cv2.rectangle(frame, closeup_one, closeup_two, (0, 0, 255), 3)

        # left and right eye horizontal length
        left_l = (landmarks.part(36).x, landmarks.part(36).y)
        left_r = (landmarks.part(39).x, landmarks.part(39).y)
        right_l = (landmarks.part(42).x, landmarks.part(42).y)
        right_r = (landmarks.part(45).x, landmarks.part(45).y)
        leftDist = distance(landmarks.part(39).x, landmarks.part(36).x, landmarks.part(39).y, landmarks.part(36).y)
        rightDist = distance(landmarks.part(45).x, landmarks.part(42).x, landmarks.part(45).y, landmarks.part(42).y)

        # left and right eye vertical length
        left_u = midpoint(landmarks.part(37), landmarks.part(38))
        left_d = midpoint(landmarks.part(41), landmarks.part(40))
        right_u = midpoint(landmarks.part(43), landmarks.part(44))
        right_d = midpoint(landmarks.part(47), landmarks.part(46))
        cv2.line(frame, left_u, left_d, (0, 255, 0), 2)
        cv2.line(frame, right_u, right_d, (0, 255, 0), 2)
        leftVerticalDistance = distance(left_d[0], left_u[0], left_d[1], left_u[1])
        rightVerticalDistance = distance(right_d[0], right_u[0], right_d[1], right_u[1])

        # left and right eye ratios are calculated to determine whether eyes are closed or open
        left_eye_ratio = (leftDist+leftVerticalDistance)/leftVerticalDistance
        right_eye_ratio = (rightDist+rightVerticalDistance)/rightVerticalDistance
        print("-------L.E.R---------")
        print(left_eye_ratio)
        print("-------R.E.R---------")
        print(right_eye_ratio)
        if left_eye_ratio > 6.0 and right_eye_ratio > 6.0:
            cv2.putText(frame, "EYES ARE CLOSED", (150, 450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
