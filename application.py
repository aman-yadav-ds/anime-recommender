from flask import Flask, render_template, request
from pipeline.prediction_pipeline import hybrid_recommendation, get_anime_recommendations
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations_frame= pd.DataFrame()

    if request.method == 'POST':
        
        try:
            input_type = request.form["input_type"]
            input_value = int(request.form["input_value"])

            if input_type == 'user_id':
                recommendations_frame = hybrid_recommendation(input_value)
            elif input_type == 'anime_id':
                recommendations_frame = get_anime_recommendations(input_value, )
        except Exception as e:
            print("Error Occured....")

    return render_template('index.html', recommendations_frame=recommendations_frame)

if __name__=='__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )