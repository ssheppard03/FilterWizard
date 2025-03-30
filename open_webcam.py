import cv2
import mediapipe as mp
import math
from obswebsocket import obsws, requests
from config import PASSWORD

DISTANCE_THRESHOLD = 0.16
DEBUG = False

max_thumb_distance = -1
max_index_distance = -1
max_middle_distance = -1
max_ring_distance = -1
max_pinky_distance = -1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

locked = False
fingers_extended = [False, False, False, False, False]

webcam = cv2.VideoCapture(1)

# Connect to OBS
ws = obsws("localhost", 4455, PASSWORD)  # Default port: 4455
ws.connect()

# Get the item ID of the "Black" source in the scene
scene_name = "WebcamScene"  # Replace with your actual scene name
response = ws.call(requests.GetSceneItemList(sceneName=scene_name))
items = response.getSceneItems()

# Find the item ID for the source named "Black"
black_item_id = None
white_item_id = None
for item in items:
    if item["sourceName"] == "Black":
        black_item_id = item["sceneItemId"]
    if item["sourceName"] == "White":
        white_item_id = item["sceneItemId"]

if black_item_id is None:
    raise ValueError("Source 'Black' not found in scene!")

if white_item_id is None:
    raise ValueError("Source 'White' not found in scene!")

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
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
                    # Access landmarks properly using the landmark attribute
                    thumb_tip = hand_landmarks.landmark[4]
                    index_finger_tip = hand_landmarks.landmark[8]
                    middle_finger_tip = hand_landmarks.landmark[12]
                    ring_finger_tip = hand_landmarks.landmark[16]
                    pinky_finger_tip = hand_landmarks.landmark[20]
                    wrist = hand_landmarks.landmark[0]
                    
                    # Calculate distance between thumb tip and wrist
                    thumb_distance = math.sqrt((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2)
                    index_distance = math.sqrt((index_finger_tip.x - wrist.x) ** 2 + (index_finger_tip.y - wrist.y) ** 2)
                    middle_distance = math.sqrt((middle_finger_tip.x - wrist.x) ** 2 + (middle_finger_tip.y - wrist.y) ** 2)
                    ring_distance = math.sqrt((ring_finger_tip.x - wrist.x) ** 2 + (ring_finger_tip.y - wrist.y) ** 2)
                    pinky_distance = math.sqrt((pinky_finger_tip.x - wrist.x) ** 2 + (pinky_finger_tip.y - wrist.y) ** 2)

                    max_thumb_distance = max(max_thumb_distance, thumb_distance)
                    max_index_distance = max(max_index_distance, index_distance)
                    max_middle_distance = max(max_middle_distance, middle_distance)
                    max_ring_distance = max(max_ring_distance, ring_distance)
                    max_pinky_distance = max(max_pinky_distance, pinky_distance)
                    
                    locked = thumb_distance < DISTANCE_THRESHOLD * max_thumb_distance and index_distance < DISTANCE_THRESHOLD * max_index_distance and middle_distance < DISTANCE_THRESHOLD * max_middle_distance and ring_distance < DISTANCE_THRESHOLD * max_ring_distance and pinky_distance < DISTANCE_THRESHOLD * max_pinky_distance

                    left_hand_visible = True

                if handedness == "Left": # Handedness is inverted, this is actually right
                    # Access landmarks properly using the landmark attribute
                    thumb_tip = hand_landmarks.landmark[4]
                    index_finger_tip = hand_landmarks.landmark[8]
                    middle_finger_tip = hand_landmarks.landmark[12]
                    ring_finger_tip = hand_landmarks.landmark[16]
                    pinky_finger_tip = hand_landmarks.landmark[20]
                    wrist = hand_landmarks.landmark[0]
                    
                    # Calculate distance between thumb tip and wrist
                    thumb_distance = math.sqrt((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2)
                    index_distance = math.sqrt((index_finger_tip.x - wrist.x) ** 2 + (index_finger_tip.y - wrist.y) ** 2)
                    middle_distance = math.sqrt((middle_finger_tip.x - wrist.x) ** 2 + (middle_finger_tip.y - wrist.y) ** 2)
                    ring_distance = math.sqrt((ring_finger_tip.x - wrist.x) ** 2 + (ring_finger_tip.y - wrist.y) ** 2)
                    pinky_distance = math.sqrt((pinky_finger_tip.x - wrist.x) ** 2 + (pinky_finger_tip.y - wrist.y) ** 2)
                    
                    fingers_extended = [
                        thumb_distance > DISTANCE_THRESHOLD,
                        index_distance > DISTANCE_THRESHOLD,
                        middle_distance > DISTANCE_THRESHOLD,
                        ring_distance > DISTANCE_THRESHOLD,
                        pinky_distance > DISTANCE_THRESHOLD
                    ]

                    right_hand_visible = True

            if DEBUG:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Toggle visibility based on fingers_extended
        if all(not finger for finger in fingers_extended) and right_hand_visible:
            # Show the "Black" source
            ws.call(requests.SetSceneItemEnabled(
                sceneName=scene_name,
                sceneItemId=black_item_id,
                sceneItemEnabled=True
            ))
        else:
            # Hide the "Black" source
            ws.call(requests.SetSceneItemEnabled(
                sceneName=scene_name,
                sceneItemId=black_item_id,
                sceneItemEnabled=False
            ))

        if all(not finger for finger in fingers_extended[1:]) and fingers_extended[0] and right_hand_visible:
            # Show the "White" source
            ws.call(requests.SetSceneItemEnabled(
                sceneName=scene_name,
                sceneItemId=white_item_id,
                sceneItemEnabled=True
            ))
        else:
            # Hide the "White" source
            ws.call(requests.SetSceneItemEnabled(
                sceneName=scene_name,
                sceneItemId=white_item_id,
                sceneItemEnabled=False
            ))
        
        if DEBUG:
            cv2.imshow("Webcam", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

ws.disconnect()
webcam.release()
cv2.destroyAllWindows()