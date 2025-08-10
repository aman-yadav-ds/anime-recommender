from config.paths_config import *
from utils.helpers import *
import joblib

try:
    anime_df = pd.read_csv(ANIME_DF)
    synopsis_df = pd.read_csv(SYNOPSIS_DF)
    rating_df = pd.read_csv(RATING_DF)
    anime_weights = joblib.load(ANIME_WEIGHTS_PATH)
    anime2anime_encoded = joblib.load(ANIME2ANIME_ENCODED)
    anime2anime_decoded = joblib.load(ANIME2ANIME_DECODED)
    user_weights = joblib.load(USER_WEIGHTS_PATH)
    user2user_encoded = joblib.load(USER2USER_ENCODED)
    user2user_decoded = joblib.load(USER2USER_DECODED)
except FileNotFoundError as e:
    raise FileNotFoundError("File not found in the given dir.", e)

def hybrid_recommendation(user_id , user_weight=0.8, content_weight =0.2):

    ## User Recommndation

    similar_users = find_similar_users(user_id,user_weights,user2user_encoded,user2user_decoded)
    user_recommended_animes =get_user_recommendations(user_id, similar_users, anime_df, synopsis_df,rating_df)
    

    user_recommended_anime_list = user_recommended_animes["anime_name"].tolist()

    #### Content recommendation
    content_recommended_animes = []

    for anime in user_recommended_anime_list:
        similar_animes = find_similar_animes(anime, anime_weights, anime2anime_encoded, anime2anime_decoded, anime_df, synopsis_df)

        if similar_animes is not None and not similar_animes.empty:
            content_recommended_animes.extend(similar_animes["eng_version"].tolist())
        else:
            print(f"No similar anime found {anime}")
    
    combined_scores = {}

    for anime in user_recommended_anime_list:
        combined_scores[anime] = combined_scores.get(anime,0) + user_weight

    for anime in content_recommended_animes:
        combined_scores[anime] = combined_scores.get(anime,0) + content_weight  

    sorted_animes = sorted(combined_scores.items() , key=lambda x:x[1] , reverse=True)

    recommended_anime_names = [anime for anime , _ in sorted_animes[:10]]

    recommended_animes_frame = getRecommendedAnimeFrame(recommended_anime_names, anime_df, synopsis_df)

    return recommended_animes_frame


def get_anime_recommendations(anime_id):
    similar_animes = find_similar_animes(anime_id, anime_weights, anime2anime_encoded, anime2anime_decoded, anime_df, synopsis_df)
    return similar_animes