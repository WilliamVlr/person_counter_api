# PERSON COUNTER API
Using YOLO v8 to detect person and count it. The endpoints can be hit to inference image or video streaming from a camera.

## HOW TO USE
1. Adjust some changes in the code if needed.
Do some changes in:
- MQTT_BROKER with broker hostname/ip
- MQTT_PORT with 1883 if broker not using TLS
- MQTT_TOPIC with the intended topic
- MQTT_USERNAME with the user's name based on access list of the broker
- MQTT_PASSWORD with the user's password
- Update the ca.crt file if there any changes from the broker's configuration or after generating new certificate
2. Add ca.crt file from your broker's signed certificate
3. Build the docker image: docker build -t atm-yolo-api .
4. Run the container: docker run -d -p 8000:8000 --name atm-detector atm-yolo-api (OR run from docker desktop GUI)
