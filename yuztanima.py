# face_recognition ve opencv kütüphanelerini import ederek başlıyoruz
import face_recognition
import cv2
import numpy as np


# opencv metodu olan VideoCapture ile webcam'den görüntü almayı başlatıyoruz // 0 default webcam video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture(0)

# Yukarıdaki "mennan sevim" resmini yüklüyoruz ve encoding bilgisini alıyoruz
serhat_image = face_recognition.load_image_file(
    "/home/serhat/Desktop/profil1.jpg")
serhat_face_encoding = face_recognition.face_encodings(serhat_image)[0]

# Yukarıdaki "miray sevim" resmini yüklüyoruz ve encoding bilgisini alıyoruz
bilinmeyen_image = face_recognition.load_image_file(
    "/home/serhat/Desktop/profil2.jpg")
bilinmeyen_face_encoding = face_recognition.face_encodings(bilinmeyen_image)[0]

# Encoding ve açıklama kısmını burada tanımlıyoruz, birden fazla tanımlayabiliriz
known_face_encodings = [serhat_face_encoding, bilinmeyen_face_encoding]
known_face_names = ["Serhat", "Yeni Kişi"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    # Videodan anlık bir kare yakalıyoruz
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Tanımlı olmayan yüzlere ne yazılacak
        name = "Bilinmiyor"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Sonuçları göster
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top += 25
            right += 10
            bottom += 15
            left += 10

        # Yüzü kırmızı çerçeve ile göster
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Yüzün bilgilerini gir
        cv2.rectangle(frame, (left, bottom + 30), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 25), font, 0.7, (255, 255, 255), 1)

    # Ekranda göster
    cv2.imshow('Yüz tanımlama sistemi', frame)

    # q tuşuna basıldığı zaman çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Kamerayı kapat
video_capture.release()
cv2.destroyAllWindows()
