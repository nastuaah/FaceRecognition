import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Подключение видеопотока, вместо 0 можно вставить ссылку на видео
cap = cv2.VideoCapture(0)

# Инициализация счетчика сжатых губ
lip_press_count = 0

# Отслеживаем были ли ранее сжаты губы, чтобы считать только новые разы сжатия
prev_lips_pressed = False

while True:
    ret, frame = cap.read()
    
    # Преобразование кадров в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Поиск лиц на фрейме
    faces = detector(gray)
    
    for face in faces:
        # Определение ориентиров на лице
        landmarks = predictor(gray, face)
        
        # Извлечение координат области рта
        mouth_points = []
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mouth_points.append((x, y))
        
        # Подсчет дистанции между верхней и нижней губой
        lip_distance = mouth_points[15][1] - mouth_points[16][1]
        
        # Если расстояние меньше порогового и губы ранее не были сжаты, увеличиваем значение счетчика
        if lip_distance < 1 and not prev_lips_pressed:
            lip_press_count += 1
            prev_lips_pressed = True
        elif lip_distance >= 1:
            prev_lips_pressed = False
    
    # Отображение фрейма
    cv2.imshow('Lip Press Detection', frame)
    
    # Завершение при нажатии q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Total lip presses: ", lip_press_count)
cap.release()
cv2.destroyAllWindows()
