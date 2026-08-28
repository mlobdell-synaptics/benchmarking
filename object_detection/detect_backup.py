import time

import cv2
from ultralytics import YOLO

# Load the lightweight YOLOv8 nano model trained on the COCO dataset
# (It will automatically download 'yolov8n.pt' on the first run)
model = YOLO("yolov8n.pt")

# Initialize the USB camera (0 is usually the default camera index)
cap = cv2.VideoCapture(0)

# Optional: Set resolution to lower values (e.g., 640x480) to boost performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

print("Starting video stream... Press 'q' to exit.")

inference_times_ms = []  # rolling log for the end-of-run summary
fps_smoothed = 0.0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    loop_start = time.perf_counter()

    # Run YOLOv8 inference on the frame
    results = model(frame, verbose=False)

    # Ultralytics reports its own preprocess/inference/postprocess split (ms/frame)
    inference_ms = results[0].speed["inference"]
    inference_times_ms.append(inference_ms)

    # Wall-clock FPS covering capture + inference + draw + display, exponentially smoothed
    loop_ms = (time.perf_counter() - loop_start) * 1000
    instant_fps = 1000 / loop_ms if loop_ms > 0 else 0.0
    fps_smoothed = instant_fps if fps_smoothed == 0 else 0.9 * fps_smoothed + 0.1 * instant_fps

    # Visualize the results on the frame (draws bounding boxes and labels)
    annotated_frame = results[0].plot()

    # Overlay timing stats
    cv2.putText(
        annotated_frame,
        f"Inference: {inference_ms:.1f} ms | FPS: {fps_smoothed:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Display the resulting frame
    cv2.imshow("YOLOv8 COCO Object Detection - Raspberry Pi 5", annotated_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

if inference_times_ms:
    avg_inference_ms = sum(inference_times_ms) / len(inference_times_ms)
    print(f"Average inference time: {avg_inference_ms:.2f} ms")
    print(f"Average inference-only FPS: {1000 / avg_inference_ms:.2f}")
    print(f"Final smoothed end-to-end FPS: {fps_smoothed:.2f}")

