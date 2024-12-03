import cv2
import numpy as np
from time import sleep

min_width = 80  # Minimum width of the rectangle
min_height = 80  # Minimum height of the rectangle

offset = 6  # Allowed error in pixels

line_position = 550  # Position of the counting line

fps_delay = 60  # FPS of the video

detected_centers = []  # List to store the centers of detected objects
vehicle_count = 0  # Counter for vehicles

# Function to calculate the center of a rectangle
def get_center(x, y, w, h):
    center_x = int(w / 2)
    center_y = int(h / 2)
    return x + center_x, y + center_y

# Load the video file
cap = cv2.VideoCapture('video.mp4')

# Initialize the background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read each frame from the video
    ret, frame = cap.read()
    frame_time = float(1 / fps_delay)  # Calculate time delay based on FPS
    sleep(frame_time)  # Sync with video frame rate
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred_frame = cv2.GaussianBlur(grayscale_frame, (3, 3), 5)  # Apply Gaussian blur
    # Apply background subtraction
    subtracted_frame = background_subtractor.apply(blurred_frame)
    # Dilate the image to fill small gaps
    dilated_frame = cv2.dilate(subtracted_frame, np.ones((5, 5)))
    # Create an elliptical kernel for morphological transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Apply morphological closing to smooth and reduce noise
    closed_frame = cv2.morphologyEx(dilated_frame, cv2.MORPH_CLOSE, kernel)
    closed_frame = cv2.morphologyEx(closed_frame, cv2.MORPH_CLOSE, kernel)
    # Find outlines (contours) of objects in the processed frame
    outlines, hierarchy = cv2.findContours(closed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame, (25, line_position), (1200, line_position), (255, 127, 0), 3)

    # Iterate through each outline
    for (i, outline) in enumerate(outlines):
        # Get the bounding rectangle for the current outline
        (x, y, w, h) = cv2.boundingRect(outline)
        # Check if the outline meets the minimum size criteria
        is_valid = (w >= min_width) and (h >= min_height)
        if not is_valid:
            continue

        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Calculate the center of the rectangle
        center = get_center(x, y, w, h)
        # Add the center to the list of detected centers
        detected_centers.append(center)
        # Draw a circle at the center of the detected object
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        # Check if the object crosses the counting line
        for (center_x, center_y) in detected_centers:
            if line_position - offset < center_y < line_position + offset:
                vehicle_count += 1  # Increment the vehicle count
                # Change the line color when a vehicle is detected
                cv2.line(frame, (25, line_position), (1200, line_position), (0, 127, 255), 3)
                # Remove the detected center to avoid counting it again
                detected_centers.remove((center_x, center_y))
                print("Vehicle detected: " + str(vehicle_count))

    # Display the vehicle count on the video
    cv2.putText(frame, "VEHICLE COUNT: " + str(vehicle_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Display the original video with overlays
    cv2.imshow("Original Video", frame)
    # Display the processed video for detection
    cv2.imshow("Detection", closed_frame)

    # Exit the loop when the ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources and close all windows
cv2.destroyAllWindows()
cap.release()
