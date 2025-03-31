import cv2
import mediapipe as mp
import math
import itertools
from obswebsocket import obsws, requests
from config import PASSWORD

class FilterWizard:

    def __init__(self):
        self.REGULAR_FINGER_THRESHOLD = 0.35  # % of palm width
        self.THUMB_THRESHOLD = 0.3           # % of palm width
        self.DEBUG = True

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.locked = False
        self.fingers_extended = [False, False, False, False, False]
        self.active_source_item_id = None

        # Create a dictionary to map finger combinations to source item IDs
        combinations = itertools.product([False, True], repeat=5)
        self.fingers_to_source_item_id = {combo: None for combo in combinations}

        # Get the item IDs
        self.sources = ["Black", "White", "Text", "Chaulk"]
        self.scene_name = "WebcamScene"  # Replace with your actual scene name

        self.ws = None

    def get_item_ids(self):
        if self.ws is None:
            raise Exception("WebSocket connection not established. Call connect_to_obs() first.")
        
        response = self.ws.call(requests.GetSceneItemList(sceneName=self.scene_name))
        items = response.getSceneItems()

        for item in items:
            for source in self.sources:
                if item["sourceName"] == source:
                    print(f"Found {source} with ID: {item['sceneItemId']}")
                    if source == "Black":
                        self.fingers_to_source_item_id[tuple([True, False, False, False, False])] = item["sceneItemId"]
                    if source == "White":
                        self.fingers_to_source_item_id[tuple([False, False, False, False, True])] = item["sceneItemId"]
                    if source == "Text":
                        self.fingers_to_source_item_id[tuple([False, False, True, False, False])] = item["sceneItemId"]
                    if source == "Chaulk":
                        self.fingers_to_source_item_id[tuple([False, True, True, False, False])] = item["sceneItemId"]

    # Add palm width calculation (for normalization)
    def get_palm_width(self, landmarks):
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        return math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)

    # Modified finger extension detection (for a single hand)
    def check_finger_extension(self, landmarks, palm_width):
        # Finger tip and MCP joint indices for each finger
        FINGER_JOINTS = {
            'thumb': {'tip': 4, 'mcp': 2},  # Different handling for thumb
            'index': {'tip': 8, 'mcp': 5},
            'middle': {'tip': 12, 'mcp': 9},
            'ring': {'tip': 16, 'mcp': 13},
            'pinky': {'tip': 20, 'mcp': 17}
        }

        extended = [False] * 5
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, name in enumerate(finger_names):
            tip = landmarks.landmark[FINGER_JOINTS[name]['tip']]
            mcp = landmarks.landmark[FINGER_JOINTS[name]['mcp']]
            
            # Calculate normalized distance
            distance = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
            normalized_distance = distance / palm_width

            # Thumb needs different handling
            if name == 'thumb':
                # Compare to middle finger's MCP for better reliability
                middle_mcp = landmarks.landmark[FINGER_JOINTS['middle']['mcp']]
                thumb_distance = math.hypot(tip.x - middle_mcp.x, tip.y - middle_mcp.y)
                extended[i] = thumb_distance / palm_width > self.THUMB_THRESHOLD
            else:
                # Other fingers use normalized distance threshold
                extended[i] = normalized_distance > self.REGULAR_FINGER_THRESHOLD

        return extended
    
    def connect_to_obs(self, host='localhost', port=4455, password=PASSWORD):
        # Connect to OBS
        self.ws = obsws("localhost", 4455, PASSWORD)  # Default port: 4455
        self.ws.connect()

    def run(self):
        self.connect_to_obs()
        if self.ws is None:
            raise Exception("WebSocket connection not established.")
        self.get_item_ids()
        with self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            try:
                # Initialize webcam, may have to iterate through the devices to find the right one
                webcam = cv2.VideoCapture(1)
                while webcam.isOpened():
                    right_hand_visible = False
                    left_hand_visible = False
                
                    success, img = webcam.read()
                    if not success:
                        continue

                    # Apply hand tracking model
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = hands.process(img_rgb)

                    # draw annotations on the image
                    if result.multi_hand_landmarks:
                        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                            # Check if the detected hand is RIGHT
                            handedness = result.multi_handedness[i].classification[0].label
                            if handedness == "Right":  # Handedness is inverted, this is actually Left
                                left_palm_width = self.get_palm_width(hand_landmarks)
                                self.locked = not all(self.check_finger_extension(hand_landmarks, left_palm_width))
                                left_hand_visible = True

                            if handedness == "Left": # Handedness is inverted, this is actually right
                                right_palm_width = self.get_palm_width(hand_landmarks)
                                self.fingers_extended = self.check_finger_extension(hand_landmarks, right_palm_width)
                                right_hand_visible = True

                        if self.DEBUG:
                            print(f"fingers: {self.fingers_extended}, locked: {self.locked}")
                            self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if self.fingers_to_source_item_id.get(tuple(self.fingers_extended)) is not None and not self.locked:
                        # Get the source item ID for the current finger combination
                        source_item_id = self.fingers_to_source_item_id[tuple(self.fingers_extended)]

                        # Enable the corresponding source item in OBS
                        self.ws.call(requests.SetSceneItemEnabled(
                            sceneName=self.scene_name,
                            sceneItemId=source_item_id,
                            sceneItemEnabled=True
                        ))
                        if self.active_source_item_id is not None and self.active_source_item_id != source_item_id:
                            # Disable the previously active source item
                            self.ws.call(requests.SetSceneItemEnabled(
                                sceneName=self.scene_name,
                                sceneItemId=self.active_source_item_id,
                                sceneItemEnabled=False
                            ))
                        self.active_source_item_id = source_item_id
                    elif not self.locked and self.active_source_item_id is not None:
                        # Disable active source item
                        self.ws.call(requests.SetSceneItemEnabled(
                            sceneName=self.scene_name,
                            sceneItemId=self.active_source_item_id,
                            sceneItemEnabled=False
                        ))
                        self.active_source_item_id = None
                    
                    
                    if self.DEBUG:
                        cv2.imshow("Webcam", img)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
            
            finally:
                self.ws.disconnect()
                webcam.release()
                cv2.destroyAllWindows()

def main():
    wizard = FilterWizard()
    wizard.run()

if __name__ == "__main__":
    main()