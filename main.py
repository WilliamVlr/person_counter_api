import time
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
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
MQTT_TOPIC = "atm/session" 

MQTT_USERNAME = "backend"
MQTT_PASSWORD = "user"
CA_CERT_PATH = "ca.crt"

HOST = "0.0.0.0"
PORT = 8000

SAVE_DIR = "./images"  # Directory to save images

# -----------------------
# SETUP DIRECTORY
# -----------------------
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# -----------------------
# APP + MODEL
# -----------------------
app = FastAPI(
    title="ATM Demo Detection API",
    description="Single Camera Demo with Image Saving",
    version="3.0"
)

print(f"Loading model {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
print("Model loaded.")

# -----------------------
# MQTT SETUP
# -----------------------
mqtt_client = None
try:
    mqtt_client = mqtt.Client(client_id="ai-yolo-demo")
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
# Simple boolean for single camera demo
session_active = False

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/person-detect", tags=["Session"])
async def detect_session(
    file: UploadFile = File(..., description="Image frame to analyze")
):
    global session_active

    # 1. Read and Decode
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode image"}

    # 2. Inference
    results = model(frame, conf=CONF_THRESHOLD,classes=[0], verbose=False)
    person_count = 0
    
    # Draw boxes on the frame for visualization
    annotated_frame = results[0].plot() 

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0: # Class 0 is Person
                person_count += 1

    # 3. Save Image locally
    timestamp = int(time.time())
    filename = f"{SAVE_DIR}/detect_{timestamp}.jpg"
    cv2.imwrite(filename, annotated_frame) # Save the frame with boxes drawn

    # 4. Session Logic (Simplified for Demo)
    mqtt_status = "idle"
    session_event = "none"

    # RULE: If person count changes to 1, start session.
    if person_count == 1:
        if not session_active:
            # --- START NEW SESSION ---
            session_event = "started"
            session_active = True
            
            payload = {
                "event": "session_start",
                "person_count": person_count,
                "timestamp": timestamp,
                "image_path": filename
            }

            if mqtt_client:
                try:
                    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                    mqtt_status = "published"
                except Exception as e:
                    mqtt_status = f"error: {e}"
        else:
            session_event = "ongoing"
    
    else:
        # --- RESET SESSION ---
        if session_active:
            session_event = "ended"
            session_active = False
        else:
            session_event = "none"

    return {
        "people": person_count,
        "session_state": session_event,
        "saved_image": filename,
        "mqtt": mqtt_status
    }

# -----------------------
# RUNNER
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)