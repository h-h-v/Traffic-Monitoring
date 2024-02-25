import cv2
import datetime

# Load pre-trained car detection classifier (Haar cascade) from local directory
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Function to detect cars in a frame
def detect_cars(frame):
    global accident_count  # Access the global accident_count variable
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw bounding boxes around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Check for accidents (overlapping bounding boxes)
    for i in range(len(cars)):
        for j in range(i+1, len(cars)):
            (x1, y1, w1, h1) = cars[i]
            (x2, y2, w2, h2) = cars[j]
            # Calculate the intersection area
            intersection_area = max(0, min(x1+w1,x2+w2) - max(x1,x2)) * max(0, min(y1+h1,y2+h2) - max(y1,y2))
            # If intersection area is greater than a threshold, consider it an accident
            if intersection_area > 1700:
                cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
    return frame, len(cars)

# Function to update and display the information at the bottom of the frame
def update_info_frame(frame, vehicle_count):
    # Get current date and time
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_day = datetime.datetime.now().strftime("%A")
    # Add current date, day, and vehicle count to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6  # Font scale
    font_thickness = 1  # Font thickness
    font_color = (255, 255, 255)
    text_offset = 10
    text_height = 20
    # Vehicle count on the left of the date
    cv2.putText(frame, f"COUNT:{vehicle_count}", (text_offset, frame.shape[0] - 3 * text_height), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    # Date on the top right corner
    date_position = (frame.shape[1] - 150, 30)
    cv2.putText(frame, f"{current_date}", date_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    # Day on the left bottom corner
    day_position = (text_offset, frame.shape[0] - text_height)
    cv2.putText(frame, f"{current_day}", day_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    return frame


# Function to process video stream
def process_video():
    # Open video capture device (webcam)
    cap = cv2.VideoCapture(0)
    # Get screen resolution
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create full screen window
    cv2.namedWindow('TRAFFIC MONITORING', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('TRAFFIC MONITORING', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Loop to continuously capture frames
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break
        # Detect cars and accidents in the frame
        frame_with_info, vehicle_count = detect_cars(frame)
        # Update and display the information at the bottom of the frame
        frame_with_info = update_info_frame(frame_with_info, vehicle_count)
        # Add Consolas font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)
        font_scale = 0.6
        font_thickness = 1
        cv2.putText(frame_with_info, "TRAFFIC MONITORING", (10, 30), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        # Display the frame with detected cars and information
        cv2.imshow('TRAFFIC MONITORING', frame_with_info)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release video capture device and close windows
    cap.release()
    cv2.destroyAllWindows()

# Start processing the video stream
process_video()