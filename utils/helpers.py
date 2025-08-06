import pandas as pd
import numpy as np
import joblib
from config.paths_config import *

############# 1. GET_ANIME_FRAME

def getAnimeFrame(anime,anime_df):
    if isinstance(anime,int):
        return anime_df[anime_df.anime_id == anime]
    if isinstance(anime,str):
        return anime_df[anime_df.eng_version == anime]
    

########## 2. GET_SYNOPSIS

def getSynopsis(id,synopsis_df):
    if isinstance(id,int):
        synopsis = synopsis_df[synopsis_df.MAL_ID == id].sypnopsis
    if isinstance(id,str):
        raise TypeError("Only pass anime_id")
    if synopsis.empty:
        return "Synopsis Not Found"
    return synopsis.values[0]

########## 3. CONTENT RECOMMENDATION

def find_similar_animes(name, path_anime_weights, path_anime2anime_encoded, path_anime2anime_decoded, path_anime_df, path_synopsis_df, n=10, return_dist=False, neg=False):
    # Load weights and encoded-decoded mappings

    anime_df = pd.read_csv(path_anime_df)
    synopsis_df = pd.read_csv(path_synopsis_df)

    anime_weights = joblib.load(path_anime_weights)
    anime2anime_encoded = joblib.load(path_anime2anime_encoded)
    anime2anime_decoded = joblib.load(path_anime2anime_decoded)

    # Get the anime ID for the given name
    index = getAnimeFrame(name, anime_df).anime_id.values[0]
    encoded_index = anime2anime_encoded.get(index)

    if encoded_index is None:
        raise ValueError(f"Encoded index not found for anime ID: {index}")

    # Compute similarity distances
    weights = anime_weights
    dists = np.dot(weights, weights[encoded_index])  # Ensure weights[encoded_index] is a 1D array
    sorted_dists = np.argsort(dists)

    n = int(n) + 1

    # Select closest or farthest based on 'neg' flag
    if neg:
        closest = sorted_dists[:n]
    else:
        closest = sorted_dists[-n:]

    # Return distances and closest indices if requested
    if return_dist:
        return dists, closest

    # Build the similarity array
    SimilarityArr = []
    for close in closest:
        decoded_id = anime2anime_decoded.get(close)
        anime_frame = getAnimeFrame(decoded_id, anime_df).iloc[0]
        synopsis = getSynopsis(int(decoded_id), synopsis_df)
        similarity = dists[close]

        row_dict = anime_frame.to_dict()
        row_dict['Synopsis'] = synopsis
        row_dict['similarity'] = similarity
        SimilarityArr.append(row_dict)
    
    Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
    Frame = Frame.drop(columns=['Type', 'Members', 'similarity'])
    return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)


######## 4. FIND_SIMILAR_USERS


def find_similar_users(user_id , path_user_weights , path_user2user_encoded , path_user2user_decoded, n=10 , return_dist=False,neg=False):
    try:

        user_weights = joblib.load(path_user_weights)
        user2user_encoded = joblib.load(path_user2user_encoded)
        user2user_decoded = joblib.load(path_user2user_decoded)

        index=user_id
        encoded_index = user2user_encoded.get(index)

        weights = user_weights

        dists = np.dot(weights,weights[encoded_index])
        sorted_dists = np.argsort(dists)

        n=n+1

        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
            

        if return_dist:
            return dists,closest
        
        SimilarityArr = []

        for close in closest:
            similarity = dists[close]

            if isinstance(user_id,int):
                decoded_id = user2user_decoded.get(close)
                SimilarityArr.append({
                    "similar_users" : decoded_id,
                    "similarity" : similarity
                })
        similar_users = pd.DataFrame(SimilarityArr).sort_values(by="similarity",ascending=False)
        similar_users = similar_users[similar_users.similar_users != user_id]
        return similar_users
    except Exception as e:
        print("Error Occured",e)


################## 5. GET USER PREF

def get_user_preferences(user_id , rating_df , anime_df ):

    animes_watched_by_user = rating_df[rating_df['user_id'] == user_id]

    user_rating_percentile = np.percentile(animes_watched_by_user['rating'] , 75)

    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user['rating'] >= user_rating_percentile]

    top_animes_user = (
        animes_watched_by_user.sort_values(by="rating" , ascending=False)['anime_id'].values
    )

    anime_df_rows = anime_df[anime_df["anime_id"].isin(top_animes_user)]
    anime_df_rows = anime_df_rows[["anime_id","eng_version","Genres"]]


    return anime_df_rows



######## 6. USER RECOMMENDATION


def get_user_recommendations(user_id, similar_users, path_anime_df , path_synopsis_df, path_rating_df, n=10):

    anime_df = pd.read_csv(path_anime_df)
    synopsis_df = pd.read_csv(path_synopsis_df)
    rating_df = pd.read_csv(path_rating_df)

    recommended_animes = []
    anime_list = []

    user_pref = get_user_preferences(int(user_id), rating_df, anime_df)
    for user_id in similar_users.similar_users.values:
        pref_list = get_user_preferences(int(user_id) , rating_df, anime_df)

        pref_list = pref_list[~pref_list.eng_version.isin(user_pref.eng_version.values)]

        if not pref_list.empty:
            anime_list.append(pref_list.eng_version.values)

    if anime_list:
            anime_list = pd.DataFrame(anime_list)

            sorted_list = pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)

            for i,anime_name in enumerate(sorted_list.index):
                n_user_pref = sorted_list[sorted_list.index == anime_name].values[0][0]

                if isinstance(anime_name,str):
                    frame = getAnimeFrame(anime_name, anime_df)
                    anime_id = frame.anime_id.values[0]
                    genre = frame.Genres.values[0]
                    synopsis = getSynopsis(int(anime_id),synopsis_df)

                    recommended_animes.append({
                        "n" : n_user_pref,
                        "MAL_ID": anime_id,
                        "anime_name" : anime_name,
                        "Genres" : genre,
                        "Synopsis": synopsis,
                    })
    return pd.DataFrame(recommended_animes).head(n)

def getRecommendedAnimeFrame(recommended_animes_names, path_anime_df, path_synopsis_df):
    anime_df = pd.read_csv(path_anime_df)
    synopsis_df = pd.read_csv(path_synopsis_df)

    recommended_anime_rows = []
    for anime_name in recommended_animes_names:
        anime_data = getAnimeFrame(anime_name, anime_df)
        anime_id = anime_data['anime_id'].values[0]
        anime_data = anime_data.iloc[0]
        synopsis = getSynopsis(int(anime_id), synopsis_df)
        
        # Create a dictionary for the new row
        row_dict = anime_data.to_dict()
        row_dict['Synopsis'] = synopsis
        recommended_anime_rows.append(row_dict)
    
    recommended_animes_frame = pd.DataFrame(recommended_anime_rows)

    recommended_animes_frame = recommended_animes_frame.drop(columns=['Type', 'Members'])

    return recommended_animes_frame