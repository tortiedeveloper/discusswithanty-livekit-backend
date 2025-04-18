from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv
import logging
import uuid
import json
import datetime
import jwt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")

if not API_KEY or not API_SECRET:
    logger.critical("LIVEKIT_API_KEY or LIVEKIT_API_SECRET not set in environment!")

@app.route('/token', methods=['POST'])
def generate_token():
    try:
        logger.debug("Request received: %s", request.data)
        data = request.json

        if data is None:
            logger.error("No JSON data in request")
            return jsonify({"error": "No JSON data provided"}), 400

        identity = data.get('identity')
        user_id = data.get('user_id')

        if not identity:
            logger.error("Missing 'identity' in request body")
            return jsonify({"error": "'identity' is required"}), 400
        if not user_id:
            logger.error("Missing 'user_id' (Firebase UID) in request body")
            return jsonify({"error": "'user_id' is required"}), 400

        unique_suffix = uuid.uuid4().hex[:12]
        room_name = f"usession-{user_id}-{unique_suffix}"
        logger.info(f"Generated UNIQUE room name: {room_name} for user_id: {user_id}")

        logger.debug(f"Generating token for identity: {identity}, room: {room_name}, user_id: {user_id}")

        now = datetime.datetime.now()
        exp = now + datetime.timedelta(hours=6)

        metadata = json.dumps({"user_id": user_id}) # Ensure metadata is a JSON string

        payload = {
            "sub": identity,
            "iss": API_KEY,
            "nbf": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "video": {
                "room": room_name,
                "roomJoin": True,
                "canPublish": True,
                "canPublishData": True,
                "canSubscribe": True
            },
            "metadata": metadata # Add the JSON string metadata here
        }

        token = jwt.encode(payload, API_SECRET, algorithm="HS256")

        # --- Added logging line ---
        logger.info(f"Generated JWT Token: {token}")
        # --------------------------

        logger.debug(f"Generated token payload (before encoding): {json.dumps(payload, indent=2)}") # Log the payload for clarity

        response = jsonify({
            "token": token,
            "room": room_name,
            "user_id": user_id
        })

        return response
    except Exception as e:
        logger.exception("Error generating token: %s", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    cert_path = 'certs/cert.pem'
    key_path = 'certs/key.pem'
    if os.path.exists(cert_path) and os.path.exists(key_path):
        logger.info(f"Running server with HTTPS on port 5000")
        app.run(host='0.0.0.0', port=5000, ssl_context=(cert_path, key_path))
    else:
        logger.warning(f"Certificates not found. Running server without HTTPS (insecure)!")
        app.run(host='0.0.0.0', port=5000)