import os
import re
import pickle
import cv2
import pytesseract
from flask import Flask, render_template, request, jsonify

# Initialize Flask App
app = Flask(_name_, template_folder="templates", static_folder="static")

# -----------------------------------------------------------------------------
# 1. Load the Machine Learning Models
# -----------------------------------------------------------------------------
diabetes_model = pickle.load(open('Saved Models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('Saved Models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('Saved Models/parkinsons_model.sav', 'rb'))

# -----------------------------------------------------------------------------
# 2. Disease-Specific Nutrient Limits
# -----------------------------------------------------------------------------
DISEASE_LIMITS = {
    "default": {
        "Total Fat": 70,
        "Saturated Fat": 20,
        "Trans Fat": 0,
        "Cholesterol": 300,
        "Sodium": 2300,
        "Dietary Fiber": 25,
        "Total Sugars": 50,
        "Added Sugars": 10,
    },
    "diabetes": {
        "Total Fat": 70,
        "Saturated Fat": 20,
        "Trans Fat": 0,
        "Cholesterol": 300,
        "Sodium": 2300,
        "Dietary Fiber": 25,
        "Total Sugars": 30,  # Stricter sugar limits
        "Added Sugars": 5,
    },
    "heart": {
        "Total Fat": 60,
        "Saturated Fat": 15,
        "Trans Fat": 0,
        "Cholesterol": 200,  # Lower for heart disease
        "Sodium": 1500,  
        "Dietary Fiber": 25,
        "Total Sugars": 50,
        "Added Sugars": 10,
    },
    "parkinsons": {
        "Total Fat": 70,
        "Saturated Fat": 20,
        "Trans Fat": 0,
        "Cholesterol": 300,
        "Sodium": 2300,
        "Dietary Fiber": 25,
        "Total Sugars": 50,
        "Added Sugars": 10,
    }
}

def get_acceptable_limits(disease: str) -> dict:
    return DISEASE_LIMITS.get(disease, DISEASE_LIMITS["default"])

# -----------------------------------------------------------------------------
# 3. OCR & Nutrition Functions
# -----------------------------------------------------------------------------
def extract_text_from_image(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(enhanced, lang="eng")

def parse_nutritional_info(text: str) -> dict:
    pattern = r"(Total Fat|Saturated Fat|Trans Fat|Cholesterol|Sodium|Dietary Fiber|Total Sugars|Added Sugars):?\s*([\d.]+)\s*(mg|g)?"
    matches = re.findall(pattern, text)
    nutrition = {}
    for nutrient, value, unit in matches:
        try:
            val = float(value)
            if unit == "mg":
                val /= 1000  # Convert mg to g
            nutrition[nutrient] = val
        except ValueError:
            continue
    return nutrition

def validate_nutrition(nutrition: dict, acceptable_limits: dict) -> dict:
    results = {}
    for nutrient, value in nutrition.items():
        if nutrient in acceptable_limits:
            limit = acceptable_limits[nutrient]
            results[nutrient] = value <= limit
    return results

# -----------------------------------------------------------------------------
# 4. Routes
# -----------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    diagnosis = None
    if request.method == 'POST':
        try:
            data = [float(request.form[key]) for key in request.form]
            prediction = diabetes_model.predict([data])
            diagnosis = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        except Exception as e:
            diagnosis = f"Error: {str(e)}"
    return render_template('diabetes.html', diagnosis=diagnosis)

@app.route('/heart-disease', methods=['GET', 'POST'])
def heart_disease():
    diagnosis = None
    if request.method == 'POST':
        try:
            data = [float(request.form[key]) for key in request.form]
            prediction = heart_disease_model.predict([data])
            diagnosis = "Has Heart Disease" if prediction[0] == 1 else "No Heart Disease"
        except Exception as e:
            diagnosis = f"Error: {str(e)}"
    return render_template('heart_disease.html', diagnosis=diagnosis)

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    diagnosis = None
    if request.method == 'POST':
        try:
            data = [float(request.form[key]) for key in request.form]
            prediction = parkinsons_model.predict([data])
            diagnosis = "Has Parkinson's" if prediction[0] == 1 else "No Parkinson's"
        except Exception as e:
            diagnosis = f"Error: {str(e)}"
    return render_template('parkinsons.html', diagnosis=diagnosis)

@app.route('/nutrition')
def nutrition_page():
    disease = request.args.get('disease', None)
    return render_template('nutrition.html', disease=disease)

@app.route('/upload', methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, image.filename)
    image.save(image_path)

    text = extract_text_from_image(image_path)
    disease = request.form.get("disease", None)
    nutrition = parse_nutritional_info(text)
    validation_results = validate_nutrition(nutrition, get_acceptable_limits(disease))

    return jsonify({
        "nutrition": nutrition,
        "validation": validation_results,
        "disease": disease
    })

# -----------------------------------------------------------------------------
# 5. Run the Application
# -----------------------------------------------------------------------------
if _name_ == '_main_':
    app.run(debug=True)