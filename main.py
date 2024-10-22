# Import necessary libraries
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears import PiGear
from libcamera import Transform
import cv2

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

    def read(self):
        # Check if stream was initialized or not
        if self.stream is None:
            return None
        # Check if we're still running
        if self.running:
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
                # Return the augmented frame
                return frame
            else:
                # Signal we're not running now
                self.running = False
        # Return None-type
        return None

    def stop(self):
        # Flag that we're not running
        self.running = False
        # Close stream
        if self.stream is not None:
            self.stream.stop()
        # Close object detector if it has a close method
        if self.object_detector is not None:
            self.object_detector.close()

# Formulate various PiCamera2 API configuration parameters
options = {
    "queue": True,
    "buffer_count": 4,
    # "controls": {"Brightness": 0.5, "ExposureValue": 2.0},
    "transform": Transform(hflip=1),
    # "sensor": {"output_size": (480, 320)},  # Will override `resolution`
    "auto_align_output_size": True,  # Auto-align output size
}

# Define class names for detection
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

# Initialize object detector
model_path = "yolov8s_h8l.hef"  # Replace with your model's path
object_detector = HailoObjectDetector(model_path=model_path, class_names=class_names, score_thresh=0.5)

# Assign your Custom Streaming Class with options to `custom_stream` attribute in options parameter
stream_options = {"custom_stream": Custom_Stream_Class(options=options, object_detector=object_detector),"custom_data_location": "/app/client"}

# Initialize WebGear_RTC app without any source
web = WebGear_RTC(enablePiCamera=True, logging=True, **stream_options)

# Run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="0.0.0.0", port=8080)

# Close app safely
web.shutdown()