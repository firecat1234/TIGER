import argparse
import cv2
import math
import mediapipe as mp
import numpy as np
import pyautogui
import toml
import os

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from typing import Any, Optional, List, Dict

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==========================
# 1) Data Structures for file-based gestures
# ==========================

class GestureDefinition:
    """
    A single gesture definition loaded from a file.
    Example fields:
      name: str
      thumb_extended: bool
      index: bool
      middle: bool
      ring: bool
      pinky: bool
      action: str (optional, for future expansions like sending keystrokes)
      thumb_rotation: Optional[str] = 'Left', 'Right', or 'Neutral'
    """
    def __init__(
        self,
        name: str,
        thumb_extended: bool,
        index: bool,
        middle: bool,
        ring: bool,
        pinky: bool,
        action: str = "",
        thumb_rotation=None
    ):
        self.name = name
        self.thumb_extended = thumb_extended
        self.index = index
        self.middle = middle
        self.ring = ring
        self.pinky = pinky
        self.action = action
        self.thumb_rotation = thumb_rotation


class GestureFileManager:
    """
    Loads and saves gesture definitions from a TOML file.
    Allows adding/removing/updating.
    """
    def __init__(self, filepath: str = "gestures.toml"):
        self.filepath = filepath
        self.gestures: List[GestureDefinition] = []
        self.load_gestures()

    def load_gestures(self) -> None:
        if not os.path.exists(self.filepath):
            print(f"No {self.filepath} found; starting with empty definitions.")
            return
        data = toml.load(self.filepath)
        for g in data.get("gesture", []):
            gd = GestureDefinition(
                name=g.get("name", "Unnamed"),
                thumb_extended=g.get("thumb_extended", False),
                index=g.get("index", False),
                middle=g.get("middle", False),
                ring=g.get("ring", False),
                pinky=g.get("pinky", False),
                action=g.get("action", ""),
                thumb_rotation=g.get("thumb_rotation", None)
            )
            self.gestures.append(gd)

    def save_gestures(self) -> None:
        data = {"gesture": []}
        for g in self.gestures:
            entry = {
                "name": g.name,
                "thumb_extended": g.thumb_extended,
                "index": g.index,
                "middle": g.middle,
                "ring": g.ring,
                "pinky": g.pinky,
                "action": g.action
            }
            if g.thumb_rotation is not None:
                entry["thumb_rotation"] = g.thumb_rotation
            data["gesture"].append(entry)
        with open(self.filepath, "w") as f:
            toml.dump(data, f)

    def add_gesture(self, gd: GestureDefinition) -> None:
        self.gestures.append(gd)
        self.save_gestures()

    def remove_gesture(self, gesture_name: str) -> None:
        self.gestures = [g for g in self.gestures if g.name != gesture_name]
        self.save_gestures()

    def update_gesture(self, gesture_name: str, updates: Dict) -> None:
        for g in self.gestures:
            if g.name == gesture_name:
                for k, v in updates.items():
                    setattr(g, k, v)
        self.save_gestures()



# ==========================
# 2) Main GestureTracker and UI code
# ==========================

class GestureTracker:
    """Handles webcam-based hand detection using MediaPipe."""

    def __init__(
        self,
        camera_index: int = 0,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        gesture_file_manager: GestureFileManager = None
    ) -> None:
        """
        Initialize the gesture tracker with camera index and detection/tracking confidences.
        Also accept a GestureFileManager to load user-defined gestures.
        """
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.running = False
        self.lock = threading.Lock()
        self.hand_detector = mp_hands.Hands(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.gesture_file_manager = gesture_file_manager  # For custom gestures

    def process_frame(self, frame: np.ndarray) -> Any:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(frame_rgb)
        return results

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> None:
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

    def classify_fingers(self, hand_landmarks: Any) -> dict:
        finger_status = {
            'thumb_extended': False,
            'thumb_rotation': 'Neutral',
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        if len(hand_landmarks.landmark) < 21:
            return finger_status

        # Thumb - tip=4, ip=2
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip  = hand_landmarks.landmark[2]
        thumb_is_extended = (thumb_tip.x < thumb_ip.x)
        finger_status['thumb_extended'] = thumb_is_extended
        if thumb_tip.x < thumb_ip.x:
            finger_status['thumb_rotation'] = 'Left'
        elif thumb_tip.x > thumb_ip.x:
            finger_status['thumb_rotation'] = 'Right'
        else:
            finger_status['thumb_rotation'] = 'Neutral'

        # Index finger - tip=8, pip=6
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        finger_status['index'] = (index_tip.y < index_pip.y)

        # Middle finger - tip=12, pip=10
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        finger_status['middle'] = (middle_tip.y < middle_pip.y)

        # Ring finger - tip=16, pip=14
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        finger_status['ring'] = (ring_tip.y < ring_pip.y)

        # Pinky - tip=20, pip=18
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        finger_status['pinky'] = (pinky_tip.y < pinky_pip.y)

        return finger_status

    def match_gesture(self, finger_map: dict) -> str:
        """
        Compare the recognized finger_map with loaded gestures from the gesture_file_manager.
        Return the name of the first matching gesture, or 'None' if none match.
        We ignore or handle thumb_rotation if needed.
        """
        if not self.gesture_file_manager:
            return "None"
        for gd in self.gesture_file_manager.gestures:
            # We'll check if extended/folded matches
            if (gd.thumb_extended == finger_map['thumb_extended'] and
                gd.index == finger_map['index'] and
                gd.middle == finger_map['middle'] and
                gd.ring == finger_map['ring'] and
                gd.pinky == finger_map['pinky']):
                # Optional check for rotation
                if gd.thumb_rotation and gd.thumb_rotation != finger_map['thumb_rotation']:
                    continue  # skip if rotation doesn't match
                return gd.name
        return "None"

    def classify_gesture(self, results: Any) -> list:
        """
        Now we rely on user-defined gestures from gestures.toml.
        We'll return a recognized gesture name for each hand.
        """
        recognized = []
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_map = self.classify_fingers(hand_landmarks)
                gesture_name = self.match_gesture(finger_map)
                recognized.append(gesture_name)
        return recognized

    def count_hands(self, results: Any) -> int:
        if results and results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks)
        return 0

    def read_frame(self) -> tuple[Optional[np.ndarray], Any]:
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        results = self.process_frame(frame)
        return frame, results

    def stop(self) -> None:
        self.cap.release()

class GestureApp(App):
    def build(self):
        self.camera_index = 0  # Default camera index
        self.detection_confidence = 0.5  # Default detection confidence
        self.tracking_confidence = 0.5  # Default tracking confidence

        self.layout = BoxLayout(orientation="vertical")

        # Webcam feed
        self.image = Image()
        self.layout.add_widget(self.image)

        # Gesture classification output
        self.gesture_label = Label(text="Gestures: None", size_hint=(1, 0.1))
        self.layout.add_widget(self.gesture_label)

        # Start/Stop button
        self.button = Button(text="Stop", size_hint=(1, 0.2))
        self.button.bind(on_press=self.toggle_camera)
        self.layout.add_widget(self.button)

        # OpenCV camera setup
        self.capture = cv2.VideoCapture(self.camera_index)
        self.hands = mp_hands.Hands(
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.is_running = True

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    def toggle_camera(self, instance):
        self.is_running = not self.is_running
        if self.is_running:
            self.button.text = "Stop"
        else:
            self.button.text = "Start"

    def update(self, dt):
        if self.is_running:
            ret, frame = self.capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                # Re-enable gesture classification
                recognized_gestures = []
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        finger_map = self.classify_fingers(hand_landmarks)
                        gesture_name = self.match_gesture(finger_map)
                        recognized_gestures.append(gesture_name)

                        # Draw landmarks on the frame
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                # Update UI label with recognized gestures
                if recognized_gestures:
                    self.gesture_label.text = f"Gestures: {', '.join(recognized_gestures)}"
                else:
                    self.gesture_label.text = "Gestures: None"

                # Flip + display
                frame = cv2.flip(frame, 0)
                buf = frame.tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
                texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
                self.image.texture = texture

    def classify_fingers(self, hand_landmarks) -> dict:
        """Determine which fingers are extended or folded."""
        finger_status = {
            'thumb_extended': False,
            'thumb_rotation': 'Neutral',
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        if len(hand_landmarks.landmark) < 21:
            return finger_status

        # Thumb - tip=4, ip=2
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip  = hand_landmarks.landmark[2]
        thumb_is_extended = (thumb_tip.x < thumb_ip.x)
        finger_status['thumb_extended'] = thumb_is_extended
        finger_status['thumb_rotation'] = 'Left' if thumb_tip.x < thumb_ip.x else 'Right' if thumb_tip.x > thumb_ip.x else 'Neutral'

        # Other fingers
        fingers = ["index", "middle", "ring", "pinky"]
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]

        for i in range(4):
            finger_status[fingers[i]] = hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[pip_ids[i]].y

        return finger_status

    def match_gesture(self, finger_map: dict) -> str:
        """Compare the recognized finger_map with loaded gestures from GestureFileManager."""
        if not hasattr(self, 'gesture_file_manager'):
            return "None"

        for gd in self.gesture_file_manager.gestures:
            if (gd.thumb_extended == finger_map['thumb_extended'] and
                gd.index == finger_map['index'] and
                gd.middle == finger_map['middle'] and
                gd.ring == finger_map['ring'] and
                gd.pinky == finger_map['pinky']):
                if gd.thumb_rotation and gd.thumb_rotation != finger_map['thumb_rotation']:
                    continue  # Skip if rotation doesn't match
                return gd.name
        return "None"


    def on_stop(self):
        self.capture.release()
        self.hands.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture Tracking Tool with UI")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the webcam to use.")
    parser.add_argument("--detection-confidence", type=float, default=0.5, help="Minimum confidence for detection.")
    parser.add_argument("--tracking-confidence", type=float, default=0.5, help="Minimum confidence for tracking.")

    args = parser.parse_args()

    # Instead of passing args to GestureApp (which Kivy does not support)
    app = GestureApp()
    app.camera_index = args.camera_index
    app.detection_confidence = args.detection_confidence
    app.tracking_confidence = args.tracking_confidence
    app.run()
