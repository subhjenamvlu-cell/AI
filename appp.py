import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🎬 AI-Based Movie Recommendation System")

st.write(
    "An AI-based movie recommendation system that recommends "
    "similar movies using Content-Based Filtering, TF-IDF "
    "and Cosine Similarity."
)

st.write("A Group Project Made By Ishita & Subhashree")


# =====================================================
# SAMPLE MOVIE DATASET
# =====================================================

data = {

    "Title": [
        "The Avengers",
        "Avengers Age of Ultron",
        "Iron Man",
        "Iron Man 2",
        "Captain America The First Avenger",
        "Captain America The Winter Soldier",
        "Thor",
        "Thor The Dark World",
        "Black Panther",
        "Doctor Strange",
        "Spider Man",
        "Spider Man Homecoming",
        "Guardians of the Galaxy",
        "Guardians of the Galaxy Vol 2",
        "Ant Man",
        "Justice League",
        "Batman Begins",
        "The Dark Knight",
        "Man of Steel",
        "Wonder Woman"
    ],

    "Genre": [
        "Action Adventure Science Fiction",
        "Action Adventure Science Fiction",
        "Action Science Fiction",
        "Action Science Fiction",
        "Action Adventure",
        "Action Adventure Thriller",
        "Action Adventure Fantasy",
        "Action Adventure Fantasy",
        "Action Adventure Science Fiction",
        "Action Adventure Fantasy",
        "Action Adventure Science Fiction",
        "Action Adventure Science Fiction Comedy",
        "Action Adventure Science Fiction Comedy",
        "Action Adventure Science Fiction Comedy",
        "Action Adventure Science Fiction Comedy",
        "Action Adventure Fantasy",
        "Action Crime Drama",
        "Action Crime Drama Thriller",
        "Action Adventure Science Fiction",
        "Action Adventure Fantasy"
    ],

    "Description": [
        "superheroes avengers save the world action team",
        "avengers superheroes fight powerful enemy action team",
        "iron man superhero technology action marvel",
        "iron man technology superhero action marvel",
        "captain america superhero war action marvel",
        "captain america winter soldier action thriller marvel",
        "thor superhero god fantasy action marvel",
        "thor fantasy superhero action adventure marvel",
        "black panther superhero wakanda action marvel",
        "doctor strange magic superhero fantasy marvel",
        "spider man superhero action adventure marvel",
        "spider man young superhero school action marvel",
        "guardians heroes space adventure superhero marvel",
        "guardians heroes space adventure comedy marvel",
        "ant man superhero technology comedy marvel",
        "superheroes justice league action adventure dc",
        "batman hero crime action drama dc",
        "batman superhero crime thriller action dc",
        "superman superhero action science fiction dc",
        "wonder woman superhero action fantasy dc"
    ]
}


df = pd.DataFrame(data)


# =====================================================
# CREATE COMBINED FEATURES
# =====================================================

df["Features"] = (
    df["Genre"] + " " +
    df["Description"]
)


# =====================================================
# TF-IDF VECTORIZATION
# =====================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

feature_matrix = vectorizer.fit_transform(
    df["Features"]
)


# =====================================================
# COSINE SIMILARITY
# =====================================================

similarity_matrix = cosine_similarity(
    feature_matrix
)


# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================

def recommend_movies(movie_name, number_of_movies=5):

    movie_index = df[
        df["Title"] == movie_name
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:number_of_movies + 1]:

        recommendations.append({
            "Movie": df.iloc[index]["Title"],
            "Genre": df.iloc[index]["Genre"],
            "Similarity": score
        })

    return pd.DataFrame(recommendations)


# =====================================================
# USER INPUT
# =====================================================

st.header("🎥 Select a Movie")

selected_movie = st.selectbox(
    "Choose a movie you like:",
    df["Title"]
)


number_of_movies = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


# =====================================================
# RECOMMENDATION BUTTON
# =====================================================

if st.button("🔍 Recommend Movies"):

    recommendations = recommend_movies(
        selected_movie,
        number_of_movies
    )

    st.subheader("🎬 Recommended Movies")

    for i, row in recommendations.iterrows():

        st.write(
            f"### {i + 1}. {row['Movie']}"
        )

        st.write(
            f"**Genre:** {row['Genre']}"
        )

        st.write(
            f"**Similarity Score:** "
            f"{row['Similarity'] * 100:.2f}%"
        )

        st.progress(
            float(row["Similarity"])
        )

        st.divider()


# =====================================================
# RECOMMENDATION CHART
# =====================================================

if st.button("📊 Show Recommendation Chart"):

    recommendations = recommend_movies(
        selected_movie,
        number_of_movies
    )

    chart_data = recommendations.set_index(
        "Movie"
    )["Similarity"]

    st.subheader(
        "📊 Movie Similarity Chart"
    )

    st.bar_chart(
        chart_data
    )


# =====================================================
# DATASET PREVIEW
# =====================================================

st.header("📁 Movie Dataset")

with st.expander("View Movie Dataset"):

    st.dataframe(
        df[
            ["Title", "Genre", "Description"]
        ],
        use_container_width=True
    )


# =====================================================
# HOW THE SYSTEM WORKS
# =====================================================

st.header("🤖 How AI Recommendation Works")

st.write(
    """
    1. The system stores movie information such as title,
       genre and description.

    2. Movie information is converted into numerical
       features using TF-IDF Vectorization.

    3. Cosine Similarity calculates how similar movies
       are to each other.

    4. Movies with the highest similarity scores are
       selected as recommendations.

    5. The recommended movies are displayed through
       the Streamlit interface.
    """
)


# =====================================================
# ABOUT PROJECT
# =====================================================

st.header("ℹ️ About This Project")

st.write(
    """
    This project uses Content-Based Filtering to recommend
    movies based on their similarity.

    TF-IDF is used to convert movie descriptions and genres
    into numerical features, while Cosine Similarity is used
    to measure similarity between movies.
    """
)


# =====================================================
# DISCLAIMER
# =====================================================

st.info(
    "ℹ️ This project is developed for educational purposes "
    "to demonstrate Artificial Intelligence and Machine "
    "Learning concepts."
)