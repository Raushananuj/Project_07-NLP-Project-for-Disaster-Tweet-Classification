from flask import Flask, render_template, request
import joblib
import pickle


# Load the model and vectorizer
model = pickle.load(open(r'logistic_regression_model.pkl','rb'))
vectorizer = pickle.load(open(r'tfidf_vectorizer.pkl','rb'))
app = Flask(__name__)

# Route for the dashboard
@app.route("/")
def home():
    return render_template('Dashboard.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("text-content")
    tokenized_text = vectorizer.transform(['text'])
    prediction = model.predict(tokenized_text)
    prediction = "Disaster tweet" if prediction ==1 else "Non Disaster tweet"
    return render_template("Dashboard.html",prediction=prediction,text=text)
    
if __name__ == "_main_":
    app.run(debug=True)
