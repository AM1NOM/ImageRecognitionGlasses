import cv2
import queue
import threading
import pyttsx3
from ultralytics import YOLO

# Load model once
model = YOLO("yolov8n.pt")

# --- TTS worker (single engine in background thread) ---
tts_queue = queue.Queue()

def tts_worker(q: queue.Queue):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    while True:
        text = q.get()
        if text is None:               # sentinel to stop
            break
        engine.say(text)
        engine.runAndWait()
        q.task_done()

tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,), daemon=True)
tts_thread.start()

# Open camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Cannot open camera")

print("Press 'p' to capture/run detection. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        # Run inference directly on the numpy frame (BGR)
        results = model(frame)       # Ultralytics accepts BGR numpy arrays
        result = results[0]

        # Annotated image (RGB) -> convert to BGR for OpenCV
        try:
            annotated = result.plot()
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow("Detections", annotated_bgr)
        except Exception:
            # plot() may fail if no image content; ignore gracefully
            pass

        # Extract boxes (may be empty)
        detections = []
        cls_tensor = getattr(result.boxes, "cls", None)
        conf_tensor = getattr(result.boxes, "conf", None)

        if cls_tensor is not None and conf_tensor is not None:
            # convert to python lists (works with torch tensors or lists)
            try:
                cls_list = cls_tensor.tolist()
                conf_list = conf_tensor.tolist()
            except Exception:
                # fallback: iterate elements
                cls_list = [int(x) for x in cls_tensor]
                conf_list = [float(x) for x in conf_tensor]

            for cls_idx, conf in zip(cls_list, conf_list):
                # class names come from model.names (not result.names)
                name = model.names[int(cls_idx)]
                detections.append({
                    "class": name,
                    "confidence": round(float(conf) * 100, 1)
                })

        # Build text to speak
        if len(detections) == 0:
            text = "No objects were detected in this image."
        elif len(detections) == 1:
            d = detections[0]
            text = f"A {d['class']} was detected with {d['confidence']} percent confidence."
        else:
            phrases = [f"{d['class']} with {d['confidence']} percent" for d in detections]
            if len(phrases) == 2:
                text = f"Multiple objects were detected: {phrases[0]} and {phrases[1]}."
            else:
                text = "Multiple objects were detected: " + ", ".join(phrases[:-1]) + ", and " + phrases[-1] + "."

        print(text)
        # enqueue for background TTS
        tts_queue.put(text)

    if key == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()

# Stop TTS thread
tts_queue.put(None)
tts_thread.join()
