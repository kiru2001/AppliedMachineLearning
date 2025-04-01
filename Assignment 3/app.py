from flask import Flask, request, render_template
import joblib
import score

app = Flask(__name__)

MODEL_FILE_PATH = "Support_Vector_Machine.joblib"

with open(MODEL_FILE_PATH, 'rb') as model_file:
    model = joblib.load(model_file)

THRESHOLD = 0.5


@app.route('/')
def home():
    """Render the home page."""
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    """Handle the spam classification request."""

    sentence = request.form['sent'] 
    label, propensity = score.score(sentence, model, THRESHOLD) 
    
    label_text = "Spam" if label == 1 else "Not spam"
    response_message = f'The sentence "{sentence}" is {label_text} with propensity {propensity}'

    return render_template('res.html', ans=response_message)


if __name__ == '__main__':
    app.run(debug=True)
