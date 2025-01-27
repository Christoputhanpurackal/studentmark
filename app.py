from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = r'models\studentmark.pkl'  # Use raw string for Windows path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Dataset
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def result():
    # Get form data
    number_courses = float(request.form['number_courses'])
    time_study = float(request.form['time_study'])

    # Prepare the input for prediction (assuming model expects two features)
    input_features = np.array([[number_courses, time_study]])

    # Predict the marks using the loaded model
    predicted_marks = model.predict(input_features)[0]  # [0] to get the first prediction result

    # Render the result page with the predicted marks
    return render_template('result.html', number_courses=number_courses, time_study=time_study, predicted_marks=predicted_marks)

if __name__ == '__main__':
    app.run(debug=True)