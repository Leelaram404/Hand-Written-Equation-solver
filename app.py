from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import re
import cv2
import os

app = Flask(__name__, template_folder="templates")

# Path to your saved model
MODEL_PATH = os.path.join(os.getcwd(), 'math_symbol_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'add', 'dec', 'div', 'eq', 'sub', 'x', 'y', 'z']

# Map predicted labels to their corresponding symbols
symbol_mapping = {
    'add': '+',
    'sub': '-',
    'div': '/',
    'eq': '=',  # You may choose to ignore the equals sign later
    'dec': '.'
}

def sanitize_equation(eq):
    """
    Clean and validate the recognized equation string.
    This function:
      - Removes unwanted characters.
      - Removes any '=' characters.
      - Collapses repeated operators.
      - Removes leading/trailing operators (except a leading minus).
      - Verifies that the final string is a valid arithmetic expression.
    
    If the expression is not valid, it returns an empty string.
    """
    # Remove any equals signs and whitespace.
    eq = eq.replace('=', '').strip()
    # Keep only allowed characters.
    allowed = "0123456789+-*/()."
    eq = "".join(ch for ch in eq if ch in allowed)
    
    # Collapse multiple consecutive operators (e.g. "++" → "+")
    eq = re.sub(r'([+\-*/])\1+', r'\1', eq)
    
    # Remove trailing operators (e.g. "1+2+" → "1+2")
    while eq and eq[-1] in "+-*/":
        eq = eq[:-1]
    
    # Remove leading operators except for a minus (allow negative numbers)
    while eq and eq[0] in "*/+":
        eq = eq[1:]
    
    # Define a regex that matches a valid arithmetic expression:
    #   - A number (integer or decimal) optionally preceded by a minus,
    #   - Followed by zero or more (operator and number) pairs.
    pattern = r'^-?\d+(\.\d+)?(?:[+\-*/]-?\d+(\.\d+)?)*$'
    if re.fullmatch(pattern, eq):
        return eq
    else:
        return ""  # Return empty if the expression is invalid

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data received"})
    
    # Remove the data URL header and decode the image
    image_data = re.sub("^data:image/.+;base64,", "", data["image"])
    pil_image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
    img_np = np.array(pil_image)
    
    # Invert the image so that strokes become white on a black background
    img_np = 255 - img_np
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours for segmentation
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
    
    # Sort bounding boxes from left to right
    boxes = sorted(boxes, key=lambda b: b[0])
    equation = ""
    print("Detected {} symbol(s)".format(len(boxes)))
    
    for (x, y, w, h) in boxes:
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(thresh.shape[1], x + w + pad), min(thresh.shape[0], y + h + pad)
        roi = thresh[y1:y2, x1:x2]
        
        # Convert ROI to a PIL image, resize to (64, 64) and convert to RGB (3 channels)
        roi_pil = Image.fromarray(roi)
        roi_pil = roi_pil.resize((64, 64))
        roi_pil = roi_pil.convert("RGB")
        
        # Convert image to numpy array, normalize, and add a batch dimension
        roi_array = np.array(roi_pil).astype("float32") / 255.0
        roi_array = np.expand_dims(roi_array, axis=0)
        
        # Predict the symbol using the model
        prediction = model.predict(roi_array)
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        # Get the proper symbol; if not in mapping, use the label directly.
        symbol = symbol_mapping.get(predicted_label, predicted_label)
        equation += symbol
    
    print("Raw recognized equation:", equation)
    sanitized = sanitize_equation(equation)
    print("Sanitized equation:", sanitized)
    
    # Evaluate the sanitized expression only if it is valid.
    if sanitized:
        try:
            result = eval(sanitized)
            equation_with_result = sanitized + " = " + str(result)
        except Exception as e:
            equation_with_result = sanitized + " [Error evaluating: {}]".format(e)
    else:
        equation_with_result = equation + " [Non-evaluable equation]"
    
    return jsonify({"prediction": equation_with_result})

if __name__ == "__main__":
    app.run(debug=True)
