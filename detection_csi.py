from picamera2 import Picamera2
import cv2
from ultralytics import YOLO


picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO11 model
model = YOLO("/home/pi/lsh-yolo/ultralytics-v11/PCSNet.pt")

# Set up video output
#output_path = "/home/pi/ultralytics/ultralytics/output/01.detection_camera_csi.mp4"
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()

    # Run YOLO11 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Write the frame to the video file
    #out.write(annotated_frame)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
picam2.close()
#out.release()
cv2.destroyAllWindows()
