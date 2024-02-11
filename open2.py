import cv2
import numpy as np

# Загрузка каскада Хаара для глаз
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Инициализация видеопотока с веб-камеры
capture = cv2.VideoCapture(0)

# Переменные состояния для отслеживания предыдущего положения глаз
previous_position = None
left_eye_counter = 0
right_eye_counter = 0

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение глаз
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    # Переменная для отслеживания текущего положения глаз
    current_position = None

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Определение позиции глаз
        eye_center_x = ex + ew // 2

        # Определение текущего положения глаз
        if eye_center_x < frame.shape[1] // 2:
            current_position = 'LEFT'
        elif eye_center_x > frame.shape[1] * 2 // 3:
            current_position = 'RIGHT'
      
    # Вывод сообщения только при смене позиции
    if current_position != previous_position:
        if current_position == 'LEFT':
            cv2.putText(frame, 'LEFT', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            left_eye_counter +=1
        
        elif current_position == 'RIGHT':
            cv2.putText(frame, 'RIGHT', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            right_eye_counter +=1
        previous_position = current_position

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f'Number of eye motions to right: {left_eye_counter}')
print(f'Number of eye motions to left: {right_eye_counter}')

capture.release()
cv2.destroyAllWindows()