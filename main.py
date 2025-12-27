import time
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import os

# CONFIG
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.85

MQTT_BROKER = "mqtt-broker.local" # change as needed (IP Broker/DNS)
MQTT_PORT = 8883 # 1883 for no TLS
MQTT_TOPIC = "atm/session" 

MQTT_USERNAME = "backend" # remove for no auth
MQTT_PASSWORD = "user" # remove for no auth
CA_CERT_PATH = "ca.crt" # remove for no TLS

HOST = "0.0.0.0"
PORT = 8000

# APP
app = FastAPI(
    title="Person Detection API",
    description="Person-Only Detection to Start Session",
    version="3.1"
)

# MODEL
print(f"Loading model {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
print("Model loaded.")

# -----------------------
# MQTT SETUP
# -----------------------
mqtt_client = None
try:
    mqtt_client = mqtt.Client(client_id="person-detection")
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD) # remove for no auth
    
    if os.path.exists(CA_CERT_PATH): # remove this if-else block for no TLS
        mqtt_client.tls_set(ca_certs=CA_CERT_PATH)
    else:
        print(f"Warning: {CA_CERT_PATH} not found. Skipping TLS for testing.")

    mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT)
    mqtt_client.loop_start()
    print("MQTT Client started.")
except Exception as e:
    print(f"MQTT Connection Error: {e}")

# STATE MANAGEMENT
session_active = False

# API ENDPOINT
@app.post("/person-detect", tags=["Session"])
async def detect_session(
    file: UploadFile = File(..., description="Image frame to analyze")
):
    global session_active

    # read file (image) and Decode using cv2
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode image"}

    # inference with filters for 'person' class only
    results = model(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
    
    # count the boxes detected (people)
    person_count = len(results[0].boxes)

    # session Logic
    mqtt_status = "idle"
    session_event = "none"

    if person_count == 1:
        if not session_active:
            # --- START SESSION ---
            session_event = "started"
            session_active = True
            timestamp = int(time.time())
            
            payload = {
                "event": "session_start",
                "person_count": person_count,
                "timestamp": timestamp,
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
        "mqtt": mqtt_status
    }

# RUN APP
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)