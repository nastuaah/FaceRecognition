import os
import cv2
import numpy as np

# Переменные для подсчета количества взглядов в каждую сторону
forward_look_count = 0
left_look_count = 0
right_look_count = 0

# Функция для определения направления взгляда
def determine_gaze_direction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    left_eye_centers = []
    right_eye_centers = []

    for (ex, ey, ew, eh) in eyes:
        eye_center_x = ex + ew // 2
        eye_center_y = ey + eh // 2

        if eye_center_x < frame.shape[1] // 2:
            left_eye_centers.append((eye_center_x, eye_center_y))
        elif eye_center_x > frame.shape[1] // 2:
            right_eye_centers.append((eye_center_x, eye_center_y))

    if len(left_eye_centers) > len(right_eye_centers):
        return 'LEFT'
    elif len(right_eye_centers) > len(left_eye_centers):
        return 'RIGHT'
    else:
        return 'FORWARD'

# Загрузка каскада Хаара для глаз
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Обработка датасета
dataset_path = 'eye_dataset'

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            gaze_direction = determine_gaze_direction(img)

            if folder == 'forward_look':
                if gaze_direction == 'FORWARD':
                    forward_look_count += 1
            elif folder == 'left_look':
                if gaze_direction == 'LEFT':
                    left_look_count += 1
            elif folder == 'right_look':
                if gaze_direction == 'RIGHT':
                    right_look_count += 1

# Вывод результатов
print(f'Number of forward looks: {forward_look_count}')
print(f'Number of left looks: {left_look_count}')
print(f'Number of right looks: {right_look_count}')