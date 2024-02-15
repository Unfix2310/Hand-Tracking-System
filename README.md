# Hand-Tracking-System
import cv2
import numpy as np
import pyautogui

# Function to detect hand
def detect_hand(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # Threshold the image to get binary image
    _, thresh = cv2.threshold(gray_blur, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (hand)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(max_contour)

        # Draw rectangle around the hand
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get centroid of the hand
        cx, cy = x + w // 2, y + h // 2

        # Move the mouse cursor based on centroid position
        screenWidth, screenHeight = pyautogui.size()
        pyautogui.moveTo(cx * screenWidth // frame.shape[1], cy * screenHeight // frame.shape[0])

    return frame

# Set up webcam
cap = cv2.VideoCapture(0)

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hand
    frame = detect_hand(frame)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
