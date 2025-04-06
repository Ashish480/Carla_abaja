import cv2

cap = cv2.VideoCapture('/dev/video10')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process the frame with UFLD
    processed_frame = process_lane_detection(frame)

    cv2.imshow("Lane Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

