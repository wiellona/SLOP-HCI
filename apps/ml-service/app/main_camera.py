import cv2
import mediapipe as mp
import time

# 1. Inisialisasi MediaPipe Holistic
# Holistic digunakan untuk mendeteksi pose tubuh, wajah, dan tangan sekaligus.
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,        # False berarti sistem memproses video stream, bukan gambar statis
    min_detection_confidence=0.5,   # Ambang batas keyakinan deteksi awal
    min_tracking_confidence=0.5     # Ambang batas keyakinan pelacakan frame berikutnya
)

# Inisialisasi modul untuk menggambar titik koordinat di layar
mp_drawing = mp.solutions.drawing_utils

# 2. Buka akses ke Webcam
# Angka '0' merepresentasikan indeks kamera bawaan laptop (webcam utama).
cap = cv2.VideoCapture(0)

print("Kamera sedang dimulai... Tekan tombol 'q' pada keyboard untuk keluar.")

while cap.isOpened():
    # Membaca setiap frame dari kamera
    success, frame = cap.read()
    if not success:
        print("Gagal menangkap frame dari kamera.")
        break

    # 3. Konversi format warna (Fundamental)
    # OpenCV membaca gambar dalam format BGR (Blue, Green, Red). 
    # MediaPipe membutuhkan format RGB. Kita harus mengonversinya.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Proses deteksi menggunakan MediaPipe
    results = holistic.process(frame_rgb)

    # 5. Logika Ekstraksi dan Output ke Terminal
    # Mengecek apakah tangan kanan terdeteksi
    if results.right_hand_landmarks:
        # Mengambil landmark pada ujung jari telunjuk (indeks ke-8 dalam topologi MediaPipe)
        index_finger_tip = results.right_hand_landmarks.landmark[8]
        
        # Output koordinat x, y, z ke terminal secara real-time
        # x dan y dinormalisasi dalam rentang [0.0, 1.0] sesuai resolusi layar
        print(f"Tangan Kanan Terdeteksi -> Telunjuk X: {index_finger_tip.x:.2f}, Y: {index_finger_tip.y:.2f}, Z: {index_finger_tip.z:.2f}")

    # 6. Menggambar titik-titik (landmarks) pada frame yang akan ditampilkan
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Menampilkan window video
    cv2.imshow('SLOP - Hand Tracking Test', frame)

    # Mekanisme untuk keluar dari loop dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Pembersihan resource
cap.release()
cv2.destroyAllWindows()
holistic.close()