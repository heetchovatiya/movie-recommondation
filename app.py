from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup 

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


def scrape_imdb_movie_details(imdb_id):
    url = f"https://www.imdb.com/title/{imdb_id}/"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Scrape the poster image
        poster = soup.find('div', class_='poster').a.img['src']
        
        # Scrape the trailer link
        trailer_section = soup.find('a', {'data-testid': 'hero-media__slate'})
        if trailer_section:
            trailer_url = f"https://www.imdb.com{trailer_section['href']}"
        else:
            trailer_url = None

        return poster, trailer_url
    else:
        return None, None
    

def get_imdb_id(movie_title, release_year):
    search_url = f"https://www.imdb.com/find?q={movie_title.replace(' ', '+')}+{release_year}&s=tt&ttype=ft"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the first search result
    result = soup.find('td', class_='result_text')
    if result and result.a:
        imdb_id = result.a['href'].split('/')[2]  # Extract IMDb ID
        return imdb_id
    return None


@app.route('/')
def home():
    # Show top 8 movies
    top_movies = movies_df.head(8)
    print(top_movies)
    return render_template('home.html', top_movies=top_movies)

@app.route('/search', methods=['GET'])
def search_movies():
    query = request.args.get('query')
    if query:
        # Filter movies based on search query
        search_results = movies_df[movies_df['Series_Title'].str.contains(query, case=False, na=False)]
        return render_template('search_results.html', query=query, movies=search_results)
    return redirect(url_for('home'))

# @app.route('/movie/<int:movie_index>')
# def movie_details(movie_index):
#     if 0 <= movie_index < len(movies_df):
#         # Get the specific movie
#         movie = movies_df.iloc[movie_index]
        
#         # Get recommendations (this should return movies with their original indices)
#         recommended_movies = get_movie_recommendations(movie_index)
        
#         # Ensure the recommended movies contain the correct Index field from the original dataset
#         return render_template('movie_details.html', movie=movie, recommended_movies=recommended_movies)
    
#     return redirect(url_for('home'))


@app.route('/movie/<int:movie_index>')
def movie_details(movie_index):
    if 0 <= movie_index < len(movies_df):
        movie = movies_df.iloc[movie_index]

        # Get IMDb ID dynamically by title and year
        imdb_id = get_imdb_id(movie['Series_Title'], movie['Released_Year'])

        # Scrape the poster and trailer using the IMDb ID
        if imdb_id:
            poster_url, trailer_url = scrape_imdb_movie_details(imdb_id)
        else:
            poster_url, trailer_url = None, None

        # Fallback to the locally stored image if the IMDb scraping fails
        if not poster_url:
            poster_url = movie['Poster_Link']

        recommended_movies = get_movie_recommendations(movie_index)

        return render_template(
            'movie_details.html',
            movie=movie,
            poster_url=poster_url,
            trailer_url=trailer_url,
            recommended_movies=recommended_movies
        )
    return redirect(url_for('home'))


# def get_movie_recommendations(movie_index):
#     sim_scores = list(enumerate(cosine_sim[movie_index]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:11]  # Exclude the movie itself
#     movie_indices = [i[0] for i in sim_scores]
#     recommended_movies = movies_df.iloc[movie_indices]
#     return recommended_movies


def get_movie_recommendations(movie_index):
    movie = movies_df.iloc[movie_index]
    recommended_movies = movies_df[
        (movies_df['Genre'] == movie['Genre']) & 
        (movies_df['IMDB_Rating'] >= movie['IMDB_Rating'])
    ].head(10)

    # Convert each row to a dictionary including the index
    recommended_movies_dict = recommended_movies.to_dict(orient='records')
    for i, movie in enumerate(recommended_movies_dict):
        movie['Index'] = recommended_movies.index[i]  # Add the original index to each movie

    return recommended_movies_dict


@app.route('/movies/<int:page>')
def movies_page(page):
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    paginated_movies = movies_df.iloc[start:end]
    return render_template('home.html', top_movies=paginated_movies)

if __name__ == '__main__':
    app.run(debug=True)
