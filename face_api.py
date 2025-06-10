from flask import Flask, request, jsonify
import face_recognition
import os
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
from bson.objectid import ObjectId

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Increase max content length
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# Validate Cloudinary configuration
cloudinary_name = os.getenv("CLOUDINARY_NAME")
cloudinary_api_key = os.getenv("CLOUDINARY_API_KEY")
cloudinary_api_secret = os.getenv("CLOUDINARY_API_SECRET")

if not all([cloudinary_name, cloudinary_api_key, cloudinary_api_secret]):
    logging.error("Missing Cloudinary credentials. Please check .env file.")
    raise ValueError("Missing Cloudinary credentials. Please check .env file.")

cloudinary.config(
    cloud_name=cloudinary_name,
    api_key=cloudinary_api_key,
    api_secret=cloudinary_api_secret,
)

# Validate MongoDB connection
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    logging.error("Missing MONGODB_URI. Please check .env file.")
    raise ValueError("Missing MONGODB_URI. Please check .env file.")

try:
    client = MongoClient(mongodb_uri)
    db = client.FaceGate
    employees = db.employees
    client.server_info()
    logging.info("MongoDB connection successful.")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise


@app.route("/recognize", methods=["POST"])
def recognize_face():
    logging.debug("Received request to /recognize")
    if "image" not in request.files:
        logging.error("No image provided in request")
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    employee_id = request.form.get("employeeId")
    logging.debug(f"Received image: {image_file.filename}, employeeId: {employee_id}")

    try:
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(image_file)
        image_url = result["secure_url"]
        public_id = result["public_id"]  # Get public_id for deletion
        logging.debug(f"Uploaded to Cloudinary: {image_url}")

        # Reset file pointer after Cloudinary upload
        image_file.seek(0)

        # Load image for face recognition
        image = face_recognition.load_image_file(image_file)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            logging.error("No faces detected in image")
            cloudinary.uploader.destroy(public_id)  # Delete uploaded image
            return jsonify({"error": "No faces detected"}), 400

        # Fetch employee by ID if provided, otherwise return Unknown
        if not employee_id:
            logging.warning("No employeeId provided")
            cloudinary.uploader.destroy(public_id)
            return jsonify(
                {
                    "employee_name": "Unknown",
                    "employee_id": None,
                    "image_url": image_url,
                }
            )

        employee = employees.find_one(
            {"_id": ObjectId(employee_id)}, {"image": 1, "name": 1}
        )
        if not employee or "image" not in employee:
            logging.warning(f"No employee found with ID: {employee_id}")
            cloudinary.uploader.destroy(public_id)
            return jsonify(
                {
                    "employee_name": "Unknown",
                    "employee_id": None,
                    "image_url": image_url,
                }
            )

        # Load employee's image from Cloudinary URL
        import requests
        from io import BytesIO

        # from bson.objectid import ObjectId

        response = requests.get(employee["image"])
        response.raise_for_status()
        emp_image = face_recognition.load_image_file(BytesIO(response.content))
        emp_encoding = face_recognition.face_encodings(emp_image)

        if not emp_encoding:
            logging.warning(
                f"No face encoding found for employee {employee.get('name', 'unknown')}"
            )
            cloudinary.uploader.destroy(public_id)
            return jsonify(
                {
                    "employee_name": "Unknown",
                    "employee_id": None,
                    "image_url": image_url,
                }
            )

        # Compare faces
        match = face_recognition.compare_faces([emp_encoding[0]], face_encodings[0])[0]
        name = employee["name"] if match else "Unknown"
        emp_id = str(employee["_id"]) if match else None

        # Delete the uploaded image from Cloudinary
        cloudinary.uploader.destroy(public_id)
        logging.debug(f"Deleted image from Cloudinary: {public_id}")

        logging.debug(f"Recognition result: {name}, ID: {emp_id}")
        return jsonify(
            {
                "employee_name": name,
                "employee_id": emp_id,
                "image_url": image_url,
            }
        )
    except Exception as e:
        logging.error(f"Error in face recognition: {str(e)}")
        cloudinary.uploader.destroy(public_id)  # Ensure deletion on error
        return jsonify({"error": f"Face recognition failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
