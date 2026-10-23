import cv2
import os

# Ensure the calib_images directory exists
os.makedirs('calib_images', exist_ok=True)

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Counter for image filenames
counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Webcam preview', frame)

    # Wait for a key press (1 ms)
    key = cv2.waitKey(1) & 0xFF

    # Save image on 's' key
    if key == ord('s'):
        filename = os.path.join('calib_images', f'calib_{counter}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        counter += 1

    # Quit on 'q' key
    if key == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()