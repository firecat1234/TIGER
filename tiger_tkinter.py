import argparse
import cv2
import math
import mediapipe as mp
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk
from typing import Any, Optional, List, Dict

# ----- For file-based gesture definitions -----
import toml
import os

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
        thumb_rotation: Optional[str] = None
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

class GestureApp:
    """A simple Tkinter-based UI for gesture tracking."""

    def __init__(
        self,
        camera_index: int = 0,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Gesture Tracker")
        self.root.geometry("1200x800")  # Default window size

        # Create a label to display frames
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Label to show recognized gestures (per hand)
        self.gesture_label = tk.Label(self.root, text="Gestures: ")
        self.gesture_label.pack()

        # Label to show the number of hands detected
        self.hand_count_label = tk.Label(self.root, text="Hands detected: 0")
        self.hand_count_label.pack()

        # Label to show finger extension states for all hands
        self.finger_info_label = tk.Label(self.root, text="Finger Info:")
        self.finger_info_label.pack()

        # Start/Stop buttons
        self.start_button = tk.Button(self.root, text="Start", command=self.start_tracking)
        self.start_button.pack(side="left", padx=5, pady=5)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_tracking, state="disabled")
        self.stop_button.pack(side="left", padx=5, pady=5)

        # ----- LOAD user-defined gestures from file -----
        self.gesture_file_manager = GestureFileManager(filepath="gestures.toml")
        # Create the tracker with the file manager
        self.tracker = GestureTracker(camera_index, detection_confidence, tracking_confidence, self.gesture_file_manager)

        self.update_delay = 10  # ms between frame updates
        self.running = False

    def start_tracking(self) -> None:
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.update_frame()

    def stop_tracking(self) -> None:
        self.running = False
        self.tracker.stop()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.video_label.config(image="")
        self.gesture_label.config(text="Gestures: ")
        self.hand_count_label.config(text="Hands detected: 0")
        self.finger_info_label.config(text="Finger Info:")

    def update_frame(self) -> None:
        if self.running:
            frame, results = self.tracker.read_frame()
            if frame is not None:
                self.tracker.draw_landmarks(frame, results)

                hand_count = self.tracker.count_hands(results)
                self.hand_count_label.config(text=f"Hands detected: {hand_count}")

                gesture_list = self.tracker.classify_gesture(results)
                self.gesture_label.config(text=f"Gestures: {gesture_list}")

                # Build a multiline string to display finger info
                finger_info_str = ""
                if results and results.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        finger_map = self.tracker.classify_fingers(hand_landmarks)
                        summary = (f"Hand {i+1}: "
                                   f"thumb={{'ext' if finger_map['thumb_extended'] else 'fold'}},"
                                   f" rot={{finger_map['thumb_rotation']}}, "
                                   f"index={{'ext' if finger_map['index'] else 'fold'}}, "
                                   f"middle={{'ext' if finger_map['middle'] else 'fold'}}, "
                                   f"ring={{'ext' if finger_map['ring'] else 'fold'}}, "
                                   f"pinky={{'ext' if finger_map['pinky'] else 'fold'}}")
                        finger_info_str += summary + "\n"

                self.finger_info_label.config(text=f"Finger Info:\n{finger_info_str}")

                # Convert to RGB for Tkinter
                cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image)
                imgtk = ImageTk.PhotoImage(image=pil_image)

                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

            self.root.after(self.update_delay, self.update_frame)

    def run(self) -> None:
        self.root.mainloop()
        self.tracker.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture Tracking Tool with UI")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the webcam to use.")
    parser.add_argument("--detection-confidence", type=float, default=0.5, help="Minimum confidence for detection.")
    parser.add_argument("--tracking-confidence", type=float, default=0.5, help="Minimum confidence for tracking.")

    args = parser.parse_args()

    app = GestureApp(
        camera_index=args.camera_index,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence
    )
    app.run()
