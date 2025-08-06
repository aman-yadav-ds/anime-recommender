from flask import Flask, render_template, request
from pipeline.prediction_pipeline import hybrid_recommendation, get_anime_recommendations
from utils.helpers import fetch_anime_posters
import pandas as pd

poster_cache = {}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations_frame= pd.DataFrame()
    error_message = None

    if request.method == 'POST':
        
        try:
            input_type = request.form["input_type"]
            input_value = int(request.form["input_value"])

            if input_type == 'user_id':
                recommendations_frame = hybrid_recommendation(input_value)
            elif input_type == 'anime_id':
                recommendations_frame = get_anime_recommendations(input_value, )

            if recommendations_frame.empty:
                error_message = "No recommendations found for this ID. Please try another."

            recommendations_frame['poster_url'] = recommendations_frame['anime_id'].apply(fetch_anime_posters)
            recommendations_frame.fillna("Not Available in DB.")
        except Exception as e:
            print("Error Occured....")

    return render_template('index.html', recommendations_frame=recommendations_frame, error_message=error_message)

if __name__=='__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )