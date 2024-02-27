from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
import time
from twilio.rest import Client
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# Twilio configuration
account_sid = 'AC3a11c9172b5e5399c1834eb12e258add'
auth_token = 'AC3a11c9172b5e5399c1834eb12e258add'
client = Client(account_sid, auth_token)

# Selenium configuration
driver = webdriver.Chrome('path_to_chrome_driver')
driver.get('https://web.whatsapp.com/')
# Ensure the user scans the QR code manually before proceeding

frequency = 2500
duration = 1000

def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

count = 0
earThresh = 0.3  # distance between vertical eye coordinate Threshold
earFrames = 48  # consecutive frames for eye closure
shapePredictor = "C:/Users/Administrator/desktop/project/shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# get the coord of left & right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        if ear < earThresh:
            count += 1

            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)

                # Make a phone call
                call = client.calls.create(
                    twiml='<Response><Say>Drowsiness detected. Please check immediately.</Say></Response>',
                    to='9443170552',
                    from_='9360165802'
                )
                print(call.sid)

                # Share location via WhatsApp
                time.sleep(5)  # Wait for 5 seconds before sending location
                input_box = driver.find_element_by_xpath('//div[@class="_2S1VP copyable-text selectable-text"][@contenteditable="true"][@data-tab="1"]')
                input_box.send_keys('Your location: latitude, longitude')
                input_box.send_keys(Keys.ENTER)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
driver.close()
