import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model("models/malaria_model.h5")

IMG_SIZE = 224

def predict_image(img_path):

    # Load image with same size used during training
    img = image.load_img(img_path, target_size=(128, 128))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)

    prob = prediction[0][0]

    # Confidence calculation
    confidence = max(prob, 1 - prob)

    # Reject images that are not blood cells
    if confidence < 0.65:
        return "Unknown Image (Not a blood cell)"

    # Classification
    if prob > 0.5:
        result = "Uninfected (No Malaria)"
        confidence = prob
    elif prob < 0.15:
        return f"Parasitized (Malaria Detected) | Confidence: {(1-prob)*100:.2f}%"
    else:
        return "Invalid Image (Not a malaria blood smear)"

    return f"{result} | Confidence: {confidence*100:.2f}%"

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            UPLOAD_FOLDER = "app/static/uploads"
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            prediction = predict_image(filepath)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)