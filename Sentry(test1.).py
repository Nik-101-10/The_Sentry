import cv2
import time
from ultralytics import YOLO
from collections import deque
import threading
import numpy as np
import os
import serial # --- Using pyserial for the TF-Luna Lidar

# --- GPIO control with a fallback ---
try:
    from gpiozero import OutputDevice, MotionSensor
    GPIO_AVAILABLE = True
except (ImportError, Exception):
    print("--- WARNING: gpiozero library not found or not on a compatible system. ---")
    print("--- PIR Sensor and Relay control will be disabled. ---")
    GPIO_AVAILABLE = False


class WebcamStream:
    """Reads frames from a webcam in a dedicated thread to prevent I/O blocking."""
    def __init__(self, src=0, width=1280, height=720):
        print("Initializing threaded webcam stream...")
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened(): raise IOError("Cannot open webcam")
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed: raise IOError("Cannot grab initial frame from webcam")
        self.latest_frame_deque = deque([self.frame], maxlen=1)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, name="WebcamStreamThread", args=())
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()
        print("Webcam stream thread started.")
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if not grabbed: self.stop()
            else: self.latest_frame_deque.append(frame)

    def read(self):
        return self.latest_frame_deque[0]

    def stop(self):
        if self.stopped: return
        self.stopped = True
        time.sleep(0.1)
        self.stream.release()
        print("Webcam stream released.")

class TFLunaLidarStream:
    """Reads data from a TF-Luna Lidar in a dedicated thread using pyserial."""
    def __init__(self, port, baudrate, offset_cm=0.0):
        print(f"Initializing TF-Luna Lidar on port {port}...")
        self.port = port
        self.baudrate = baudrate
        self.offset_m = offset_cm / 100.0
        self.ser = None
        self.distance_m = float('inf')
        self.stopped = False

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print("TF-Luna serial port opened successfully.")
        except serial.SerialException as e:
            raise IOError(f"Failed to open TF-Luna serial port {port}: {e}")

        self.thread = threading.Thread(target=self.update, name="TFLunaLidarThread", args=())
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()
        print("TF-Luna Lidar thread started.")
        return self

    def update(self):
        print("TF-Luna Lidar update() thread has entered its loop.")
        while not self.stopped:
            if self.ser.in_waiting >= 9:
                if self.ser.read(1) == b'\x59' and self.ser.read(1) == b'\x59':
                    data_frame = self.ser.read(7)
                    raw_distance_cm = data_frame[0] + data_frame[1] * 256
                    self.distance_m = (raw_distance_cm / 100.0) + self.offset_m
            else:
                time.sleep(0.01)

    def read_distance(self):
        return self.distance_m

    def stop(self):
        if self.stopped: return
        self.stopped = True
        print("Stopping TF-Luna Lidar stream...")
        time.sleep(0.1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("TF-Luna serial port closed.")


class SentrySystem:
    # --- PERFORMANCE TUNING CONSTANTS ---
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    PROCESS_EVERY_N_FRAMES = 5
    YOLO_IMG_SIZE = 320
    YOLO_CONFIDENCE = 0.3
    YOLO_MODEL_PATH = 'yolov8n.pt'

    # --- STATE MACHINE AND TIMEOUT CONSTANTS ---
    VERIFICATION_TIMEOUT = 5.0
    ACTIVE_COOLDOWN = 10.0

    # --- TF-LUNA LIDAR CONFIGURATION ---
    LIDAR_PORT = '/dev/ttyUSB0'
    LIDAR_BAUDRATE = 115200
    LIDAR_DISTANCE_OFFSET_CM = 0.0
    LIDAR_ACTIVATION_DISTANCE_METERS = 3.0

    # --- PHOTO CAPTURE CONFIGURATION ---
    PHOTO_SAVE_DIR = "captures"

    def __init__(self):
        print(f"Initializing Sentry with resolution: {self.FRAME_WIDTH}x{self.FRAME_HEIGHT}")
        self.yolo_model = YOLO(self.YOLO_MODEL_PATH)
        
        self.state = "IDLE"
        self.last_activity_time = 0
        self.window_is_open = False

        self.stream = WebcamStream(src=0, width=self.FRAME_WIDTH, height=self.FRAME_HEIGHT).start()
        time.sleep(1.0) 

        os.makedirs(self.PHOTO_SAVE_DIR, exist_ok=True)
        print(f"Photos will be saved to '{self.PHOTO_SAVE_DIR}/' directory.")

        initial_frame = self.stream.read()
        self.frame_height, self.frame_width, _ = initial_frame.shape
        self.zone_width = self.frame_width // 3
        self.trail = deque(maxlen=50)
        
        self.frame_counter = 0
        self.last_known_boxes = []
        self.last_known_cat_in_action_zone = False
        self.last_known_distance = float('inf')

        self.photo_taken_for_this_event = False

        self.fps_deque = deque(maxlen=30)
        self.start_time = time.time()
        
        self.relay = None
        self.pir_sensor = None
        self.relay_is_on = False
        if GPIO_AVAILABLE:
            try:
                self.relay_pin = 23
                self.relay = OutputDevice(self.relay_pin, active_high=False, initial_value=False)
                self.pir_pin = 18 
                self.pir_sensor = MotionSensor(self.pir_pin)
                self.pir_sensor.when_motion = self.handle_pir_trigger
                print("Relay and PIR Sensor initialized.")
            except Exception as e:
                print(f"--- WARNING: Failed to initialize GPIO. Error: {e} ---")
                self.relay, self.pir_sensor = None, None
        else:
             print("--- Running in simulation mode (no GPIO). Press 't' to simulate a PIR trigger. ---")

        self.lidar_stream = None
        try:
            self.lidar_stream = TFLunaLidarStream(
                port=self.LIDAR_PORT,
                baudrate=self.LIDAR_BAUDRATE,
                offset_cm=self.LIDAR_DISTANCE_OFFSET_CM
            ).start()
        except Exception as e:
            print(f"--- FATAL: Could not initialize TF-Luna Lidar stream. Error: {e} ---")
            self.lidar_stream = None

    def handle_pir_trigger(self):
        if self.state == "IDLE":
            print("\nMOTION DETECTED! Waking up system to verify target...")
            self.state = "VERIFYING"
            self.last_activity_time = time.time()
        elif self.state in ["VERIFYING", "ACTIVE"]:
            self.last_activity_time = time.time()

    # --- NEW DRAWING LOGIC: SEPARATED FOR CLEAN PHOTOS ---

    def draw_detection_info(self, frame, boxes):
        """Draws only the bounding boxes and confidence scores for detections."""
        for box_info in boxes:
            bbox, conf, zone = box_info['bbox'], box_info['confidence'], box_info['zone']
            color = (0, 0, 255) if zone == 'action' else (0, 255, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f'Cat {conf:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_live_ui(self, frame):
        """Draws non-persistent UI elements for the live feed only."""
        for i in range(1, 3):
            cv2.line(frame, (i * self.zone_width, 0), (i * self.zone_width, self.frame_height), (255, 0, 0), 2)
        cv2.putText(frame, "ACTION", (self.zone_width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.fps_deque.append(time.time() - self.start_time)
        self.start_time = time.time()
        fps = len(self.fps_deque) / sum(self.fps_deque) if self.fps_deque else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (self.frame_width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        dist_color = (0, 255, 0) if self.last_known_distance > self.LIDAR_ACTIVATION_DISTANCE_METERS else (0, 0, 255)
        distance_text = f"Dist: {self.last_known_distance:.2f}m" if self.last_known_distance != float('inf') else "Dist: N/A"
        cv2.putText(frame, distance_text, (self.frame_width - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dist_color, 2)

        status_text = f"STATE: {self.state}"; status_color = (0, 255, 255)
        if self.state == "ACTIVE":
            status_text = "STATUS: RELAY ACTIVE" if self.relay_is_on else "STATUS: CAT DETECTED"
            status_color = (0, 0, 255) if self.relay_is_on else (0, 255, 0)
        cv2.putText(frame, status_text, (20, self.frame_height - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)

    def take_photo(self, clean_frame, distance, boxes):
        """Creates and saves a photo with specific overlays, leaving the original frame untouched."""
        output_frame = clean_frame.copy()
        
        # 1. Draw the essential detection info (bounding boxes)
        self.draw_detection_info(output_frame, boxes)

        # 2. Draw the "Target Eliminated" overlay text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color_red = (0, 0, 255)
        text_color_white = (255, 255, 255)
        
        status_text = "TARGET ELIMINATED"
        date_text = f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        location_text = "Location: Action Zone"
        distance_text = f"Distance: {distance:.2f} m"

        cv2.putText(output_frame, status_text,     (20, 40), font, 1.1, text_color_red, 2, cv2.LINE_AA)
        cv2.putText(output_frame, date_text,       (20, 80), font, 0.7, text_color_white, 2, cv2.LINE_AA)
        cv2.putText(output_frame, location_text,   (20, 110), font, 0.7, text_color_white, 2, cv2.LINE_AA)
        cv2.putText(output_frame, distance_text,   (20, 140), font, 0.7, text_color_white, 2, cv2.LINE_AA)

        # 3. Generate filename and save the composed image
        timestamp_filename = time.strftime("%Y%m%d_%H%M%S")
        dist_str = f"{distance:.2f}m".replace('.', 'p')
        filename = os.path.join(self.PHOTO_SAVE_DIR, f"capture_{timestamp_filename}_dist-{dist_str}.jpg")
        
        try:
            cv2.imwrite(filename, output_frame)
            print(f"*** PHOTO TAKEN: Saved as {filename} with clean overlays ***")
        except Exception as e:
            print(f"--- ERROR: Could not save photo to {filename}. Reason: {e} ---")

    def get_zone(self, bbox):
        x1, _, x2, _ = bbox
        center_x = int((x1 + x2) / 2)
        return 'action' if self.zone_width <= center_x < 2 * self.zone_width else 'watch'

    def draw_trail(self, frame):
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                if self.trail[i - 1] is None or self.trail[i] is None: continue
                cv2.line(frame, self.trail[i - 1], self.trail[i], (0, 255, 255), 3)
            if self.trail[-1]:
                cv2.circle(frame, self.trail[-1], 8, (0, 0, 255), -1)

    def run(self):
        print("Sentry System starting. Press 'q' to quit.")
        
        while True:
            if self.state == "IDLE":
                if self.window_is_open: cv2.destroyAllWindows(); self.window_is_open = False
                print(f"\rCurrent State: IDLE (Waiting for PIR trigger...)", end="")
                time.sleep(0.2)
                key = cv2.waitKey(1) & 0xFF
                if not GPIO_AVAILABLE and key == ord('t'): self.handle_pir_trigger()
                if key == ord('q'): break
                continue

            frame = self.stream.read()
            if frame is None: break

            if self.lidar_stream: self.last_known_distance = self.lidar_stream.read_distance()

            cat_was_detected_this_frame = False
            if self.frame_counter % self.PROCESS_EVERY_N_FRAMES == 0:
                results = self.yolo_model(frame, verbose=False, conf=self.YOLO_CONFIDENCE, classes=[15], imgsz=self.YOLO_IMG_SIZE)
                current_boxes, cat_in_action_zone_now = [], False
                if results and results[0].boxes:
                    cat_was_detected_this_frame = True
                    self.trail.clear()
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        bbox, zone = (x1, y1, x2, y2), self.get_zone((x1, y1, x2, y2))
                        if zone == 'action': cat_in_action_zone_now = True
                        current_boxes.append({'bbox': bbox, 'confidence': float(box.conf[0]), 'zone': zone})
                        self.trail.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                self.last_known_boxes, self.last_known_cat_in_action_zone = current_boxes, cat_in_action_zone_now
            
            if self.state == "VERIFYING":
                if cat_was_detected_this_frame:
                    print("\nCAT CONFIRMED! Switching to ACTIVE mode."); self.state = "ACTIVE"; self.last_activity_time = time.time()
                elif time.time() - self.last_activity_time > self.VERIFICATION_TIMEOUT:
                    print("\nVerification timed out. Returning to IDLE."); self.state = "IDLE"; self.trail.clear()
                    continue
            elif self.state == "ACTIVE":
                if cat_was_detected_this_frame: self.last_activity_time = time.time() 
                elif time.time() - self.last_activity_time > self.ACTIVE_COOLDOWN:
                    print("\nCooldown expired. Returning to IDLE."); self.state = "IDLE"; self.trail.clear()
                    if self.relay and self.relay_is_on: self.relay.off()
                    self.relay_is_on = False; self.photo_taken_for_this_event = False
                    continue

                is_close_enough = self.last_known_distance < self.LIDAR_ACTIVATION_DISTANCE_METERS
                is_in_hot_zone = self.last_known_cat_in_action_zone and is_close_enough

                if is_in_hot_zone:
                    if not self.relay_is_on:
                        print(f"\nCAT IN HOT ZONE ({self.last_known_distance:.2f}m) - ACTIVATING SYSTEMS!")
                        if self.relay: self.relay.on()
                        self.relay_is_on = True
                        if not self.photo_taken_for_this_event:
                            self.take_photo(frame, self.last_known_distance, self.last_known_boxes)
                            self.photo_taken_for_this_event = True
                else:
                    if self.relay_is_on:
                        print("\nTarget left hot zone - DEACTIVATING RELAY.")
                        if self.relay: self.relay.off()
                        self.relay_is_on = False
            
            # --- UPDATED DRAWING ORDER ---
            self.draw_detection_info(frame, self.last_known_boxes)
            if self.trail: self.draw_trail(frame)
            self.draw_live_ui(frame)

            cv2.imshow('Sentry System', frame)
            self.window_is_open = True

            self.frame_counter += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.stop()

    def stop(self):
        print("\nStopping Sentry System...")
        if self.stream: self.stream.stop()
        if self.lidar_stream: self.lidar_stream.stop()
        if GPIO_AVAILABLE: 
            if self.relay: self.relay.close()
            if self.pir_sensor: self.pir_sensor.close()
            print("GPIO resources released.")
        cv2.destroyAllWindows()
        print("Sentry System stopped.")


if __name__ == "__main__":
    sentry_instance = None
    try:
        sentry_instance = SentrySystem()
        sentry_instance.run()
    except (IOError, serial.SerialException) as e:
        print(f"\n--- FATAL ERROR during initialization or run: {e} ---")
    except KeyboardInterrupt:
        print("\nUser interrupted. Shutting down.")
    finally:
        if sentry_instance:
            sentry_instance.stop()
