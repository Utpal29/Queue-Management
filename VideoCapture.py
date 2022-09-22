import cv2 as cv
import face_detection

detector = face_detection.build_detector('RetinaNetMobileNetV1', confidence_threshold=0.5)
cap = cv.VideoCapture('queue.mp4')

def detect_faces(image):
    return detector.detect(image)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = detect_faces(frame)
    face_id = {}
    index = 1
    for face in faces:
        x1, y1, x2, y2, confidence = face
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        tracker = cv.TrackerKCF_create()
        tracker.init(frame, bbox)
        face_id[index] = tracker
        index = index + 1
    for index in face_id:
        tracker = face_id[index]
        ok, bbox = tracker.update(frame)
        if ok:
            x1, y1, w, h = [int(v) for v in bbox]
            cv.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
            cv.putText(frame, f'ID:{index}', (x1, y1-20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cv.imshow('Cam Feed', frame)
    cv.waitKey(1)
