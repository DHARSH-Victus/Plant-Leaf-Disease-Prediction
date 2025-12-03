# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the pre-trained model
filepath = 'E:/jeeva project/Mini1/Plant-Leaf-Disease-Prediction-main/model.h5'
model = load_model(filepath)
print(model)
print("Model Loaded Successfully")


# Prediction function
def pred_tomato_dieas(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))  # Load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # Convert image to numpy array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimensions from 3D to 4D

    result = model.predict(test_image)  # Predict the disease
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)  # Get the predicted class
    print(pred)

    # Map predictions to class names and corresponding HTML files
    if pred == 0:
        return "Guava - Bacterial Affect", 'Guava - Bacterial_affect.html'
    elif pred == 1:
        return "Papaya - Viral Affect", 'Papaya - Viral_affect.html'
    elif pred == 2:
        return "Tomato - Bacterial Spot Disease", 'Tomato-Bacteria Spot.html'
    elif pred == 3:
        return "Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'
    elif pred == 4:
        return "Tomato - Healthy and Fresh", 'Tomato-Healthy.html'
    elif pred == 5:
        return "Tomato - Late Blight Disease", 'Tomato - Late_blight.html'
    elif pred == 6:
        return "Tomato - Leaf Mold Disease", 'Tomato - Leaf_Mold.html'
    elif pred == 7:
        return "Tomato - Septoria Leaf Spot Disease", 'Tomato - Septoria_leaf_spot.html'
    elif pred == 8:
        return "Tomato - Target Spot Disease", 'Tomato - Target_Spot.html'
    elif pred == 9:
        return "Tomato - Tomato Yellow Leaf Curl Virus Disease", 'Tomato - Tomato_Yellow_Leaf_Curl_Virus.html'
    elif pred == 10:
        return "Tomato - Tomato Mosaic Virus Disease", 'Tomato - Tomato_mosaic_virus.html'
    elif pred == 11:
        return "Tomato - Two Spotted Spider Mite Disease", 'Tomato - Two-spotted_spider_mite.html'
    else:
        return "Unknown Disease", 'Error.html'


# Create Flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# Get input image from client, then predict class and render respective .html page for solution
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['image']  # Fetch input
            filename = file.filename
            print("@@ Input posted = ", filename)

            file_path = os.path.join('E:/jeeva project/Mini1/Plant-Leaf-Disease-Prediction-main/static/upload/', filename)
            file.save(file_path)

            print("@@ Predicting class......")
            result = pred_tomato_dieas(tomato_plant=file_path)

            # Ensure the result is valid
            if result is None or len(result) != 2:
                return "Prediction failed. Please try again.", 500

            pred, output_page = result
            return render_template(output_page, pred_output=pred, user_image=filename)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "An error occurred during prediction. Please check logs.", 500


# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=100100)
