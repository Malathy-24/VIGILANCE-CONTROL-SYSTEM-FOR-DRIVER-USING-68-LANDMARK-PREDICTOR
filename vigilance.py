from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
# Twilio credentials
account_sid = "your id"
auth_token = "your twilio token"
twilio_phone_number = "your twilio number"
emergency_contact_number = "your mob number"
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def send_alert_message():
    client = Client(account_sid, auth_token)
    try:
         message = client.messages.create( to=emergency_contact_number,from_=twilio_phone_number,body="Emergency! Please check the location, Tracking URL: https://maps.app.goo.gl/xfrgGqw4zJy1DRSr8",)
         print(f"Alert message sent to {emergency_contact_number}. Message SID: {message.sid}")
    except Exception as e:
        print(f"Error sending alert message: {str(e)}")

def make_emergency_call():
    client = Client(account_sid, auth_token)

    try:
        response = VoiceResponse()
        response.say("This is an emergency call. The location has been shared. Please help!!")
        call = client.calls.create(
            to=emergency_contact_number,
            from_=twilio_phone_number,
            twiml=str(response)
        )

        print(f"Emergency call initiated to {emergency_contact_number}. Call SID: {call.sid}")
    except Exception as e:
        print(f"Error making emergency call: {str(e)}")
frequency = 2500
duration = 1000
def blinked(a, b, c, d, e, f):
    up = dist.euclidean(b, d) + dist.euclidean(c, e)
    down = dist.euclidean(a, f)
    ratio = up / (2.0 * down)
# Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                winsound.Beep(frequency, duration)

                # After 5 seconds, trigger the alert message and emergency call
                cv2.waitKey(5000)  # Wait for 5 seconds
                send_alert_message()
                make_emergency_call()

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy :("
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Result of detector", face_frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
