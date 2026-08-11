import os
import joblib
import numpy as np
from flask import Flask, request, render_template_string

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('wine_quality_model.pkl')

# HTML template for the web interface
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Wine Quality Prediction</title>
<form action="/predict" method="post">
    <p>Fixed Acidity: <input type="number" name="fixed_acidity" required></p>
    <p>Volatile Acidity: <input type="number" name="volatile_acidity" required></p>
    <p>Citric Acid: <input type="number" name="citric_acid" required></p>
    <p>Residual Sugar: <input type="number" name="residual_sugar" required></p>
    <p>Chlorides: <input type="number" name="chlorides" required></p>
    <p>Free Sulfur Dioxide: <input type="number" name="free_sulfur_dioxide" required></p>
    <p>Total Sulfur Dioxide: <input type="number" name="total_sulfur_dioxide" required></p>
    <p>Density: <input type="number" name="density" required></p>
    <p>pH: <input type="number" name="pH" required></p>
    <p>Sulphates: <input type="number" name="sulphates" required></p>
    <p>Alcohol: <input type="number" name="alcohol" required></p>
    <button type="submit">Predict Quality</button>
</form>

{% if prediction %}
    <h3>Predicted Wine Quality: {{ prediction }}</h3>
{% endif %}

</body>
</html>
"""

# Route for the home page
@app.route('/')
def home():
    return render_template_string(HTML_FORM, prediction=None)

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    features = [
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
        float(request.form['density']),
        float(request.form['pH']),
        float(request.form['sulphates']),
        float(request.form['alcohol'])
    ]
    
    # Make prediction
    input_array = np.array[features]
    prediction = model.predict(input_array)[0]

    # Render the result on the web page
    return render_template_string(HTML_FORM, prediction=round(prediction, 2))

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
  