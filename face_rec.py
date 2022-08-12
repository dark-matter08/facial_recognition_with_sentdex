# pip install face_recognition
import encodings
import face_recognition
import os
import cv2
import pickle
import time


KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6

FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" #hog

# VIDEO = cv2.VideoCapture(0)
VIDEO = cv2.VideoCapture("facerecvideo.mp4")

print("========================== loading known faces ==========================")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    try:
        print(f"==========> {name}")
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
            # image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
            # encoding = face_recognition.face_encodings(image)[0]
            encoding = pickle.load(open(f"{name}/{filename}", "rb"))
            known_faces.append(encoding)
            known_names.append(name)

    except Exception as e:
        print(e)

if len(known_names) > 0:
    next_id = max(known_names) + 1

else:
    next_id = 0


print("======================== processing unknown faces =======================")
while True:
    # image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    ret, image = VIDEO.read()
    locations = face_recognition.face_locations(image, model=MODEL)
    encoding = face_recognition.face_encodings(image, locations)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encoding, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match Found: {match}")
        else:
            match = f"{next_id}"
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f"{KNOWN_FACES_DIR}/{match}")
            the_time = time.time()
            the_time = str(the_time)
            pickle.dump(face_encoding, open(f"{KNOWN_FACES_DIR}/{match}/{match}-{the_time}.pkl", "wb"))

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        color = [0, 255, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)

        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

        cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # cv2.waitKey(0)
    # cv2.destroyWindow(filename)
