import cv2
import dlib
import np

# Load the pre-trained face landmark detection model from dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Capture video from the webcam
cap = cv2.VideoCapture("video3.mp4")

# Initialize the lip press counter
lip_press_count = 0

# Flag to track if lips were previously pressed
prev_lips_pressed = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector(gray)
    
    for face in faces:
        # Detect the facial landmarks
        landmarks = predictor(gray, face)
        
        # Extract the coordinates of the mouth region
        mouth_points = []
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mouth_points.append((x, y))
        
        # Calculate the distance between upper and lower lip
        lip_distance = mouth_points[15][1] - mouth_points[16][1]
        
        # If the distance is less than a threshold and lips were not previously pressed, increment the counter
        if lip_distance < 1 and not prev_lips_pressed:
            lip_press_count += 1
            prev_lips_pressed = True
        elif lip_distance >= 1:
            prev_lips_pressed = False
        
        # Draw a rectangle around the mouth region
        x, y, w, h = cv2.boundingRect(np.array(mouth_points))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the frame with annotations
    cv2.imshow('Lip Press Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
print("Total lip presses: ", lip_press_count)
cap.release()
cv2.destroyAllWindows()
