from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load IMDb Top 1000 Movies Data
file_path = 'imdb_top_1000.csv'  # Path to your CSV file
movies_df = pd.read_csv(file_path)

# Preprocess and prepare data
movies_df.fillna('', inplace=True)
movies_df['combined_features'] = movies_df.apply(lambda row: row['Genre'] + " " + row['Director'] + " " + row['Star1'] + " " + row['Star2'] + " " + row['Star3'] + " " + row['Star4'], axis=1)

# Use TF-IDF to convert text into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/')
def home():
    # Show top 8 movies
    top_movies = movies_df.head(8)
    print(top_movies)
    return render_template('home.html', top_movies=top_movies)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if query:
        matching_movies = movies_df[movies_df['Series_Title'].str.contains(query, case=False, na=False)]
        return render_template('home.html', top_movies=matching_movies)
    return redirect(url_for('home'))

@app.route('/movie/<int:movie_index>')
def movie_details(movie_index):
    if 0 <= movie_index < len(movies_df):
        movie = movies_df.iloc[movie_index]
        recommended_movies = get_movie_recommendations(movie_index)
        return render_template('movie_details.html', movie=movie, recommended_movies=recommended_movies)
    return redirect(url_for('home'))

def get_movie_recommendations(movie_index):
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the movie itself
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = movies_df.iloc[movie_indices]
    return recommended_movies

@app.route('/movies/<int:page>')
def movies_page(page):
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    paginated_movies = movies_df.iloc[start:end]
    return render_template('home.html', top_movies=paginated_movies)

if __name__ == '__main__':
    app.run(debug=True)
