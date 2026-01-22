from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO
import numpy as np


# Initialize Flask

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})


# Load Keras model

MODEL_PATH = "/Users/ahmedkhader/Desktop/project skin/Best_skin_disease_Acc98%.h5"
model = load_model(MODEL_PATH)

# Class names
class_names = ["Acne", "Benign Keratosis", "Eczema", "Psoriasis"]


# Prepare image

def read_image(file_bytes) -> np.ndarray:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Common/general treatments

common_treatments = [
    "Keep the affected skin clean and dry",
    "Avoid scratching or irritating the area",
    "Use gentle, fragrance-free skincare products",
    "Apply moisturizer daily to prevent dryness",
    "Avoid prolonged direct sunlight exposure",
    "Consult a dermatologist if symptoms worsen"
]


# Disease-specific treatments

specific_treatments = {
    "Acne": [
        "Use salicylic acid or benzoyl peroxide products",
        "Avoid oily skincare products",
        "Wash face twice daily with a mild cleanser",
        "Consider retinoid creams for persistent acne"
    ],
    "Benign Keratosis": [
        "Usually harmless and may not require treatment",
        "Moisturizing creams can reduce dryness",
        "Dermatologist can remove lesions if needed"
    ],
    "Eczema": [
        "Use fragrance-free moisturizers regularly",
        "Avoid hot showers and harsh soaps",
        "Topical corticosteroid creams for flare-ups",
        "Keep skin hydrated throughout the day"
    ],
    "Psoriasis": [
        "Use medicated moisturizers to reduce scaling",
        "Phototherapy for chronic cases",
        "Topical steroid or vitamin D creams",
        "Avoid stress and skin triggers when possible"
    ]
}


# Home page

@app.route("/")
def home():
    return render_template("home.html")


# Predict API

@app.route("/predict", methods=["POST"])
def predict():
    try:
        uploaded_file = request.files.get("file")
        if uploaded_file is None or uploaded_file.filename == "":
            return render_template("error.html", error="⚠️ Please upload a skin image before submitting.")

        # Read and preprocess image
        img_bytes = uploaded_file.read()
        img_array = read_image(img_bytes)

        # Make prediction
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        prediction = class_names[class_idx]

        # Get specific treatments for predicted class
        disease_specific = specific_treatments.get(prediction, ["No specific treatment available"])

        return render_template(
            "result.html",
            prediction=prediction,
            probability=confidence,
            common_treatments=common_treatments,
            specific_treatments=disease_specific
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)