import cv2 as cv 
import numpy as np 

capture = cv.VideoCapture(0) #сюда можно добавить ссылку на видео вместо видеопотока
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml") 

# Инициализация переменных для отслеживания наклона головы и счетчика наклонов головы
angle = 0
head_tilt = False  # Initialize head tilt state
head_tilt_count = 0

while True: 
    ret, frame = capture.read() 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Находиv лица на изображении с помощью классификатора лиц
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) 
    x, y, w, h = 0, 0, 0, 0

    # Отрисовываем прямоугольники вокруг обнаруженных лиц на кадре
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Находим глаза в области лица с помощью классификатора глаз
    eyes = eye_cascade.detectMultiScale(gray[y:(y + h), x:(x + w)], 1.1, 4)
    index = 0
    eye_1 = [None, None, None, None]
    eye_2 = [None, None, None, None]

    # Отрисовываем прямоугольники вокруг обнаруженных глаз на кадре
    for (ex, ey, ew, eh) in eyes:
      if index == 0:
          eye_1 = [ex, ey, ew, eh]
      elif index == 1:
          eye_2 = [ex, ey, ew, eh] 
      cv.rectangle(frame[y:(y + h), x:(x + w)], (ex, ey), 
         (ex + ew, ey + eh), (0, 0, 255), 2) 
      index = index + 1
      
    # Проверяем, что оба глаза были обнаружены
    if (eye_1[0] is not None) and (eye_2[0] is not None):
    # Определяеv, какой глаз находится слева, а какой справа    
      if eye_1[0] < eye_2[0]:
          left_eye = eye_1
          right_eye = eye_2 
      else:
          left_eye = eye_2
          right_eye = eye_1
          
      left_eye_center = ( 
      int(left_eye[0] + (left_eye[2] / 2)), 
      int(left_eye[1] + (left_eye[3] / 2))) 

      right_eye_center = ( 
      int(right_eye[0] + (right_eye[2] / 2)), 
      int(right_eye[1] + (right_eye[3] / 2))) 

      left_eye_x = left_eye_center[0] 
      left_eye_y = left_eye_center[1] 
      right_eye_x = right_eye_center[0] 
      right_eye_y = right_eye_center[1] 

      # Вычисляем разницу между координатами глаз
      delta_x = right_eye_x - left_eye_x 
      delta_y = right_eye_y - left_eye_y 

      if delta_x != 0:
      # Вычисляем угол наклона головы
        angle = np.arctan(delta_y / delta_x)
        
      else:
        angle = 0  

    angle = (angle * 180) / np.pi
    

    # Увеличиваем счетчик наклонов головы, если угол наклона больше 30 градусов
    if angle > 30:
        cv.putText(frame, 'RIGHT TILT :' + str(int(angle))+' degrees',
                    (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2, cv.LINE_4)
        if not head_tilt:
            head_tilt = True 
            head_tilt_count += 1  
    elif angle < -30:
        cv.putText(frame, 'LEFT TILT :' + str(int(angle))+' degrees',
                    (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2, cv.LINE_4)
        if not head_tilt:
            head_tilt = True  
            head_tilt_count += 1  
    else:
        cv.putText(frame, 'STRAIGHT :', (20, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2, cv.LINE_4)
        head_tilt = False  # Сбрасываем переменную состояния наклона головы в false
      

    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print(f'Number of head tilts: {head_tilt_count}')


capture.release() 
cv.destroyAllWindows()
