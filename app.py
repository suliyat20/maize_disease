from flask import Flask, request, render_template, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import sys
# Initialize the Flask app
app = Flask(__name__)

# Load your model
model = load_model('./ResNet50.keras')
# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(260, 260))  # Resize the image to the model input size
    img = image.img_to_array(img)  # Convert to a NumPy array
    img = np.expand_dims(img, axis=0)  # Add a batch dimension (1, 260, 260, 3)
    img = img / 255.0  # Normalize the image (as was done during training)
    return img


# Define route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define class labels based on your model's output
labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Define route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Create the 'uploads' folder if it doesn't exist
        upload_folder = os.path.join('static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded file
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Preprocess the uploaded image
    img = preprocess_image(file_path)

    # Perform the prediction using the loaded model
    predictions = model.predict(img)[0]
    Blight, Common_Rust, Gray_Leaf_Spot, Healthy = predictions

    has_maize_disease = (
        Blight > 0.5
        or Common_Rust > 0.5
        or Gray_Leaf_Spot > 0.5
    )
    maize_disease_value = max(Blight, Common_Rust, Gray_Leaf_Spot, Healthy)
    print("Raw predictions:", predictions)
    predicted_class = np.argmax(predictions)
    print("Predicted class index:", predicted_class)
    print("Predicted class label:", labels[predicted_class])
    print("All class probabilities:", maize_disease_value)

    # Map the prediction to class labels
    confidence_threshold = 0.5
    # if np.max(predictions) > confidence_threshold:
    #     result = labels[has_maize_disease]
    if has_maize_disease:
        print("reach here:", has_maize_disease)
        result = labels[predicted_class]
    else:
        result = "Uncertain prediction"
    # Render the result.html template, passing the prediction result and image path
    return render_template('result.html', image_path=file_path, result=result)
    # print("Softmax output:", predictions)
    # print("Sum of probabilities:", np.sum(predictions))

# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
