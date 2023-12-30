import cv2

# Load mô hình nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mở kết nối với camera (đối số 0 là camera mặc định)
cap = cv2.VideoCapture(0)

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()

    # Chuyển đổi frame sang ảnh xám để tăng tốc độ xử lý
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Vẽ hình vuông bao quanh khuôn mặt và đoán biểu cảm
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        
    # Hiển thị frame với khuôn mặt được nhận diện và biểu cảm
    cv2.imshow('Nhu con cac', frame)

    # Nhấn phím 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng kết nối với camera và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
