from flask import Flask, request, jsonify, render_template
from app.ocr_processor import extract_text_from_image
from app.validation import parse_nutritional_info, validate_nutrition

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def home():
    """
    Serve the main HTML page.
    """
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Endpoint for uploading an image and getting validation results.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    image_path = f"uploads/{image.filename}"
    image.save(image_path)

    # Extract and process text
    text = extract_text_from_image(image_path)
    nutrition = parse_nutritional_info(text)
    validation_results = validate_nutrition(nutrition)

    return jsonify({
        "nutrition": nutrition,
        "validation": validation_results,
    })
