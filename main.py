import time
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import os

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.4

MQTT_BROKER = "mqtt-broker.local"  # Set to 'localhost' for local testing
MQTT_PORT = 8883
MQTT_TOPIC = "atm/session" # Updated topic

MQTT_USERNAME = "backend"
MQTT_PASSWORD = "user"
CA_CERT_PATH = "ca.crt"

HOST = "0.0.0.0"
PORT = 8000

# -----------------------
# APP + MODEL
# -----------------------
app = FastAPI(
    title="ATM Session Detection API",
    description="YOLOv8 Detection to Trigger ATM Sessions",
    version="2.0"
)

print(f"Loading model {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
print("Model loaded.")

# -----------------------
# MQTT SETUP
# -----------------------
mqtt_client = None
try:
    mqtt_client = mqtt.Client(client_id="ai-yolo-session")
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    if os.path.exists(CA_CERT_PATH):
        mqtt_client.tls_set(ca_certs=CA_CERT_PATH)
    else:
        print(f"Warning: {CA_CERT_PATH} not found. Skipping TLS for testing.")

    mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT)
    mqtt_client.loop_start()
    print("MQTT Client started.")
except Exception as e:
    print(f"MQTT Connection Error: {e}")

# -----------------------
# STATE MANAGEMENT
# -----------------------
# Stores boolean state: { "camera_01": True/False }
# True = Session is currently active
# False/None = No session
camera_sessions = {}

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/detect/session", tags=["Session"])
async def detect_session(
    camera_id: str = Form(..., description="ID of the camera source"),
    file: UploadFile = File(..., description="Image frame to analyze")
):
    # 1. Read and Decode
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode image"}

    # 2. Inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    person_count = 0
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0: # Class 0 is Person
                person_count += 1

    # 3. Session Logic
    is_active = camera_sessions.get(camera_id, False)
    mqtt_status = "idle"
    session_event = "none"

    # RULE: If person count changes to 1, start session.
    if person_count == 1:
        if not is_active:
            # --- START NEW SESSION ---
            session_event = "started"
            camera_sessions[camera_id] = True # Set flag to active
            
            payload = {
                "camera_id": camera_id,
                "event": "session_start",
                "person_count": person_count,
                "timestamp": int(time.time())
            }

            if mqtt_client:
                try:
                    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                    mqtt_status = "published"
                except Exception as e:
                    mqtt_status = f"error: {e}"
        else:
            # --- SESSION ALREADY ACTIVE ---
            # Do nothing, session continues
            session_event = "ongoing"
    
    else:
        # --- RESET SESSION ---
        # If person count is 0 (empty) or > 1 (crowd), end the session state
        # so it can trigger again next time it becomes exactly 1.
        if is_active:
            session_event = "ended" # Optional: You could publish a "session_end" here if needed
            camera_sessions[camera_id] = False
        else:
            session_event = "none"

    return {
        "camera_id": camera_id,
        "people": person_count,
        "session_state": session_event,
        "mqtt": mqtt_status
    }

# -----------------------
# RUNNER
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)