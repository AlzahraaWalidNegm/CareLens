import re

# Nutritional limits (in grams/milligrams)
ACCEPTABLE_LIMITS = {
    "Total Fat": 70,
    "Saturated Fat": 20,
    "Trans Fat": 0,
    "Cholesterol": 300,
    "Sodium": 2300,
    "Dietary Fiber": 25,
    "Total Sugars": 50,
    "Added Sugars": 10,
}

def parse_nutritional_info(text: str) -> dict:
    """
    Parse nutritional information from text using regex.

    Args:
        text (str): Extracted text from OCR.

    Returns:
        dict: Nutritional values as a dictionary.
    """
    pattern = r"(Total Fat|Saturated Fat|Trans Fat|Cholesterol|Sodium|Dietary Fiber|Total Sugars|Added Sugars):?\s*([\d.]+)\s*(mg|g)?"
    matches = re.findall(pattern, text)

    # Convert matches to a dictionary
    nutrition = {}
    for nutrient, value, unit in matches:
        value = float(value)
        if unit == "mg":
            value /= 1000  # Convert milligrams to grams
        nutrition[nutrient] = value

    return nutrition

def validate_nutrition(nutrition: dict) -> dict:
    """
    Validate extracted nutritional values against predefined limits.

    Args:
        nutrition (dict): Nutritional values.

    Returns:
        dict: Validation results.
    """
    results = {}
    for nutrient, value in nutrition.items():
        if nutrient in ACCEPTABLE_LIMITS:
            limit = ACCEPTABLE_LIMITS[nutrient]
            results[nutrient] = value <= limit

    return results
