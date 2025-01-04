from flask import Flask, render_template, request, jsonify
from sentimentAnalyzer.pipeline.prediction import PredictionPipeline

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")


@app.route("/process_data", methods=['POST'])
def process_data():
    user_input = request.form.get('tweets', '')

    # Analyzing Sentiment of the User
    model = PredictionPipeline()
    sentiment = model.predict(user_input)    
    
    return jsonify({'message': sentiment})

if __name__ == "__main__":
    app.run(debug=True)
              