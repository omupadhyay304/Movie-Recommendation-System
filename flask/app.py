import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
import re

# Initialize the Flask app
app = Flask(__name__)

# Load the datasets
movies_data = pd.read_csv("movies.csv")
ratings_data = pd.read_csv("ratings.csv")

# Preprocess the data
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies_data['genres_list'] = movies_data['genres'].str.replace('|', ' ')
movies_data['clean_title'] = movies_data['title'].apply(clean_title)

ratings_data = ratings_data.drop(['timestamp'], axis=1)
combined_data = ratings_data.merge(movies_data, on='movieId')

# Vectorizer for titles
vectorizer_title = TfidfVectorizer(ngram_range=(1, 2))
tfidf_title = vectorizer_title.fit_transform(movies_data['clean_title'])

# Vectorizer for genres
vectorizer_genres = TfidfVectorizer(ngram_range=(1, 2))
tfidf_genres = vectorizer_genres.fit_transform(movies_data['genres_list'])

# Function to search movies by title
def search_by_title(title):
    title = clean_title(title)
    query_vec = vectorizer_title.transform([title])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies_data.iloc[indices][::-1]
    return results

# Function to search movies by similar genres
def search_similar_genres(genres):
    query_vec = vectorizer_genres.transform([genres])
    similarity = cosine_similarity(query_vec, tfidf_genres).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = movies_data.iloc[indices][::-1]
    return results

# Function to calculate recommendation scores
def scores_calculator(movie_id):
    similar_users = combined_data[(combined_data['movieId'] == movie_id) & (combined_data['rating'] >= 4)]['userId'].unique()
    similar_user_recs = combined_data[(combined_data['userId'].isin(similar_users)) & (combined_data['rating'] >= 4)]['movieId']
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    all_users = combined_data[(combined_data['movieId'].isin(similar_user_recs.index)) & (combined_data['rating'] >= 4)]
    all_users_recs = all_users['movieId'].value_counts() / all_users['userId'].nunique()

    genres_of_selected_movie = combined_data[combined_data['movieId'] == movie_id]['genres_list'].unique()
    genres_of_selected_movie = np.array2string(genres_of_selected_movie)
    movies_with_similar_genres = search_similar_genres(genres_of_selected_movie)

    indices = []
    for index in movies_with_similar_genres[(movies_with_similar_genres['movieId'].isin(similar_user_recs.index))]['movieId']:
        indices.append(index)
    similar_user_recs.loc[indices] = similar_user_recs.loc[indices] * 1.5

    indices = []
    for index in movies_with_similar_genres[(movies_with_similar_genres['movieId'].isin(all_users_recs.index))]['movieId']:
        indices.append(index)
    all_users_recs.loc[indices] = all_users_recs.loc[indices] * 0.9

    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ['similar', 'all']
    rec_percentages['score'] = rec_percentages['similar'] / rec_percentages['all']

    rec_percentages = rec_percentages.sort_values('score', ascending=False)
    return rec_percentages

# Function to get recommendation results
def recommendation_results(user_input, title=0):
    title_candidates = search_by_title(user_input)
    movie_id = title_candidates.iloc[title]['movieId']
    scores = scores_calculator(movie_id)
    results = scores.head(10).merge(movies_data, left_index=True, right_on='movieId')[['movieId', 'score', 'clean_title', 'genres_list']]
    results = results.rename(columns={'clean_title': 'title', 'genres_list': 'genres'})
    return results

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Route to handle the main page and form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('movie_title')
        title_choice = int(request.form.get('title_choice', 0))
        recommendations = recommendation_results(user_input, title_choice)
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'))
    else:
        return render_template('index.html', recommendations=None)

# Route to get similar movie titles based on input
@app.route('/titles', methods=['POST'])
def get_titles():
    user_input = request.form.get('movie_title')
    titles = search_by_title(user_input)['clean_title'].tolist()
    return jsonify({'titles': titles})

if __name__ == '__main__':
    app.run(debug=True)

