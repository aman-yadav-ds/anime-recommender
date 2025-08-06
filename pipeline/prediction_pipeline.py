from config.paths_config import *
from utils.helpers import *

def hybrid_recommendation(user_id , user_weight=0.8, content_weight =0.2):

    ## User Recommndation

    similar_users = find_similar_users(user_id,USER_WEIGHTS_PATH,USER2USER_ENCODED,USER2USER_DECODED)
    user_recommended_animes =get_user_recommendations(user_id, similar_users, ANIME_DF, SYNOPSIS_DF,RATING_DF)
    

    user_recommended_anime_list = user_recommended_animes["anime_name"].tolist()

    #### Content recommendation
    content_recommended_animes = []

    for anime in user_recommended_anime_list:
        similar_animes = find_similar_animes(anime, ANIME_WEIGHTS_PATH, ANIME2ANIME_ENCODED, ANIME2ANIME_DECODED, ANIME_DF, SYNOPSIS_DF)

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

    recommended_animes_frame = getRecommendedAnimeFrame(recommended_anime_names, ANIME_DF, SYNOPSIS_DF)

    return recommended_animes_frame


def get_anime_recommendations(anime_id):
    similar_animes = find_similar_animes(anime_id, ANIME_WEIGHTS_PATH, ANIME2ANIME_ENCODED, ANIME2ANIME_DECODED, ANIME_DF, SYNOPSIS_DF)
    return similar_animes
