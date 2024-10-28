# Import necessary libraries
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears import PiGear
from libcamera import Transform
import cv2
import re
import threading
import time
import os

# Define an interface for object detectors
class ObjectDetector:
    def __init__(self):
        pass

    def detect(self, frame, frame_w, frame_h):
        """
        Detect objects in the frame.

        Args:
            frame: The frame to run detection on.
            frame_w: Width of the original frame.
            frame_h: Height of the original frame.

        Returns:
            detections: a list of detections, where each detection is a tuple:
                (class_name, bbox, score)
        """
        pass

    def close(self):
        pass

# Hailo object detector implementation
class HailoObjectDetector(ObjectDetector):
    def __init__(self, model_path, class_names, score_thresh=0.5):
        super().__init__()
        self.model_path = model_path
        self.class_names = class_names
        self.score_thresh = score_thresh
        # Initialize Hailo model here
        self.hailo_model = self.initialize_hailo_model()
        self.model_h, self.model_w, _ = self.hailo_model.get_input_shape()

    def initialize_hailo_model(self):
        from picamera2.devices import Hailo
        # Initialize Hailo model
        hailo_model = Hailo(self.model_path)
        return hailo_model

    def detect(self, frame, frame_w, frame_h):
        # Resize frame to the model input size
        inference_frame = cv2.resize(frame, (self.model_w, self.model_h))
        # Run inference
        results = self.hailo_model.run(inference_frame)
        # Extract detections
        detections = self.extract_detections(results[0], frame_w, frame_h)
        return detections

    def extract_detections(self, hailo_output, w, h):
        """Extract detections from the HailoRT-postprocess output."""
        results = []
        for class_id, detections in enumerate(hailo_output):
            for detection in detections:
                score = detection[4]
                if score >= self.score_thresh:
                    y0, x0, y1, x1 = detection[:4]
                    bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                    results.append([self.class_names[class_id], bbox, score])
        return results

    def close(self):
        # Close Hailo model if needed
        self.hailo_model.close()

# OpenCV face detector implementation
class OpenCVFaceDetector(ObjectDetector):
    def __init__(self):
        super().__init__()
        # Load the Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print(f"Error loading Haar Cascade XML file for face detection from {haarcascade_path}.")

    def detect(self, frame, frame_w, frame_h):
        # Convert frame to grayscale as Haar Cascade works on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Prepare detections in the format [(class_name, bbox, score), ...]
        detections = []
        for (x, y, w, h) in faces:
            bbox = (x, y, x + w, y + h)
            detections.append(('face', bbox, 1.0))  # Assuming score=1.0 for detected faces
        return detections

    def close(self):
        pass  # No special cleanup required

# Define your own custom streaming class
class Custom_Stream_Class:
    """
    Custom Streaming using PiGear with object detection
    """

    def __init__(self, options={}, object_detector=None):
        # Initialize PiGear with the provided options
        self.stream = PiGear(camera_num=0, logging=True, **options).start()
        # Define running flag
        self.running = True
        # Set the object detector
        self.object_detector = object_detector

        # Read screen resolution from /sys/class/graphics/fb0/modes
        self.fb_width, self.fb_height = self.get_framebuffer_resolution()
        # Open framebuffer device
        self.fbdev = '/dev/fb0'  # or '/dev/fb1' depending on your system
        try:
            self.fbfd = open(self.fbdev, 'wb')  # Open in write binary mode
        except IOError as e:
            print(f"Error opening framebuffer device {self.fbdev}: {e}")
            self.fbfd = None

        # Thread-safe variable to hold the latest frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Start the inference and framebuffer writing thread
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

    def get_framebuffer_resolution(self):
        try:
            with open('/sys/class/graphics/fb0/modes', 'r') as f:
                mode_line = f.readline().strip()
                # Extract the resolution
                match = re.search(r'(\d+)x(\d+)', mode_line)
                if match:
                    fb_width = int(match.group(1))
                    fb_height = int(match.group(2))
                else:
                    fb_width, fb_height = 800, 480
        except IOError as e:
            print(f"Error reading framebuffer modes: {e}")
            fb_width, fb_height = 800, 480  # Default resolution
        return fb_width, fb_height

    def inference_loop(self):
        while self.running:
            # Read frame from the stream
            frame = self.stream.read()
            # Check if frame is available
            if frame is not None:
                # Get frame dimensions
                frame_h, frame_w = frame.shape[:2]
                # If object_detector is set, run detection
                if self.object_detector is not None:
                    # Run detection
                    detections = self.object_detector.detect(frame, frame_w, frame_h)
                    # Draw detections on the frame
                    if detections:
                        for class_name, bbox, score in detections:
                            x0, y0, x1, y1 = bbox
                            label = f"{class_name} %{int(score * 100)}"
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x0 + 5, y0 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Write frame to framebuffer
                self.write_to_framebuffer(frame)
                # Update the latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            else:
                # End the loop if no frame is available
                self.running = False
            # Add a small delay to avoid high CPU usage
            time.sleep(0.01)

    def write_to_framebuffer(self, frame):
        if self.fbfd is None:
            return
        # Convert frame to BGRA
        frame32 = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # Resize frame to framebuffer resolution
        fbframe = cv2.resize(frame32, (self.fb_width, self.fb_height))
        # Write to framebuffer
        try:
            self.fbfd.seek(0)
            self.fbfd.write(fbframe.tobytes())
            self.fbfd.flush()
        except IOError as e:
            print(f"Error writing to framebuffer: {e}")

    def read(self):
        # Return the latest frame
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                return None

    def stop(self):
        # Flag that we're not running
        self.running = False
        # Wait for the inference thread to finish
        if self.inference_thread.is_alive():
            self.inference_thread.join()
        # Close stream
        if self.stream is not None:
            self.stream.stop()
        # Close object detector if it has a close method
        if self.object_detector is not None:
            self.object_detector.close()
        # Close framebuffer device
        if self.fbfd is not None:
            self.fbfd.close()

# Formulate various PiCamera2 API configuration parameters
options = {
    "queue": True,
    "buffer_count": 4,
    # "controls": {"Brightness": 0.5, "ExposureValue": 2.0},
    "transform": Transform(hflip=1),
    # "sensor": {"output_size": (480, 320)},  # Will override `resolution`
    "auto_align_output_size": True,  # Auto-align output size
}

# Define class names for detection (used only if Hailo is available)
class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
               'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
               'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet',
               'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Check if Hailo AI accelerator is available
if os.path.exists('/dev/hailo0'):
    # Hailo device exists, use HailoObjectDetector
    print("Hailo AI accelerator detected. Using HailoObjectDetector.")
    model_path = "yolov8s_h8l.hef"  # Replace with your model's path
    object_detector = HailoObjectDetector(model_path=model_path, class_names=class_names, score_thresh=0.5)
else:
    # Hailo device not found, use OpenCVFaceDetector
    print("Hailo AI accelerator not detected. Using OpenCVFaceDetector.")
    object_detector = OpenCVFaceDetector()

# Assign your Custom Streaming Class with options to `custom_stream` attribute in options parameter
stream_options = {
    "custom_stream": Custom_Stream_Class(options=options, object_detector=object_detector),
    "custom_data_location": "/app/client"
}

# Initialize WebGear_RTC app without any source
web = WebGear_RTC(enablePiCamera=True, logging=True, **stream_options)

# Run this app on Uvicorn server at address http://localhost:8080/
uvicorn.run(web(), host="0.0.0.0", port=8080)

# Close app safely
web.shutdown()