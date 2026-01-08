import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from model import train_and_save, load_model, recommend_movies, get_movie_info

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_LOGO_BASE = "https://image.tmdb.org/t/p/w92"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

NO_POSTER_URL = "https://placehold.co/342x513/1a1a2e/ffffff?text=No+Poster"

@st.cache_data(ttl=86400)
def get_watch_providers(movie_id: int, country: str = "US") -> dict:
    """
    Fetch streaming availability from TMDB.
    
    Returns dict with 'flatrate' (subscription), 'rent', and 'buy' options.
    Each contains list of {provider_name, logo_path}.
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
    params = {"api_key": TMDB_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Get results for specified country
        country_data = data.get("results", {}).get(country, {})
        
        result = {"flatrate": [], "rent": [], "buy": [], "link": country_data.get("link")}
        
        for provider_type in ["flatrate", "rent", "buy"]:
            for provider in country_data.get(provider_type, []):
                result[provider_type].append({
                    "name": provider["provider_name"],
                    "logo": TMDB_LOGO_BASE + provider["logo_path"] if provider.get("logo_path") else None
                })
        
        return result
    except Exception as e:
        print(f"Error fetching providers: {e}")
        return {"flatrate": [], "rent": [], "buy": [], "link": None}


def display_watch_providers(movie_id: int):
    """Display streaming options for a movie."""
    providers = get_watch_providers(int(movie_id))
    if providers["flatrate"]:
        st.write("**Stream on:**")
        
        # Build HTML as a single clean string (no f-string multiline issues)
        img_tags = ""
        for provider in providers["flatrate"][:5]:
            if provider["logo"]:
                img_tags += f"<img src='{provider['logo']}' width='35' style='border-radius:6px; margin-right:6px;'/>"
        
        # Wrap in a flex container
        html = f"<div style='display:flex; align-items:center;'>{img_tags}</div>"
        st.markdown(html, unsafe_allow_html=True)
        
        # # Names as compact caption
        # names = ", ".join([p["name"] for p in providers["flatrate"][:5]])
        # st.caption(names)
    
    elif providers["rent"]:
        st.write("**Rent on:**")
        names = ", ".join([p["name"] for p in providers["rent"][:3]])
        st.caption(names)
    
    else:
        st.caption("Streaming info not available")
    
    if providers["link"]:
        st.markdown(
            f"<a href='{providers['link']}' style='font-size: 12px; color: #888;'>All watch options →</a>",
            unsafe_allow_html=True
        )
    
@st.cache_resource  # This decorator caches the model across all users/reruns
def load_recommender():
    """Load the trained model and data. Cached to avoid reloading on every interaction."""
    if not os.path.exists("model_artifacts"):
        train_and_save()
    model, feature_data, data, encoders = load_model()
    return model, feature_data, data

# Load on startup
model, feature_data, data = load_recommender()

def get_poster_url(poster_path):
    if pd.notna(poster_path) and str(poster_path).strip():
        return TMDB_IMAGE_BASE + str(poster_path)
    return NO_POSTER_URL

def format_budget(budget):
    if budget >= 1_000_000:
        return f"${budget / 1_000_000:.0f}M"
    elif budget >= 1_000:
        return f"${budget / 1_000:.0f}K"
    return f"${budget:,.0f}"

# STREAMLIT UI

# Page config
st.set_page_config(
    page_title="GoWatch",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("GoWatch: A Movie Recommender")
st.write("Find similar movies based on genres, plot, keywords, and more!")

# Sidebar controls
st.sidebar.header("Settings")

# Movie selection - searchable dropdown
movie_list = sorted(data['title'].tolist())
selected_movie = st.sidebar.selectbox(
    "Choose a movie you like:",
    options=movie_list,
    index=movie_list.index("Inception") if "Inception" in movie_list else 0
)

# Number of recommendations slider
n_recommendations = st.sidebar.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=15,
    value=6
)

# Budget filter (optional)
st.sidebar.subheader("Filter by Budget")
budget_filter = st.sidebar.radio(
    "Show only:",
    options=["All Movies", "Indie (<\\$30M)", "Mid (\\$30M-\\$100M)", "Blockbuster (>\\$100M)"]
)

genres_list = data['genres'].fillna('').apply(
    lambda x: [g.strip() for g in x.split(',') if g.strip()]
)
genre_set = set()
for genre_list in genres_list:
    for genre in genre_list:
        genre_set.add(genre)

st.sidebar.subheader("Filter by Genre")
genre_filter = st.sidebar.radio(
    "Show only:",
    options = ["All Genres"] + sorted(list(genre_set))
)

# MAIN CONTENT

# Show selected movie info
st.subheader("Your Selection")
input_movie = get_movie_info(selected_movie, data)

if input_movie:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image(get_poster_url(input_movie['poster_path']), width=200)
    
    with col2:
        st.markdown(f"### {input_movie['title']}")
        st.markdown(f"**Rating:** {input_movie["vote_average"]:.1f}")
        st.write(f"**Genres:** {input_movie['genres']}")
        st.write(f"**Budget:** {format_budget(input_movie['budget'])}")
        st.write(f"**Overview:** {input_movie['overview']}" )
        display_watch_providers(input_movie["id"])

st.divider()

# Get recommendations
st.subheader(f"Movies Similar to '{selected_movie}'")

recommendations = recommend_movies(
    movie_title=selected_movie,
    data=data,
    feature_data=feature_data,
    model=model,
    n_recommendations=n_recommendations + 10  # Get extras in case we filter some out
)

if recommendations:
    # Apply budget filter
    if budget_filter == "Indie (<\\$30M)":
        recommendations = [r for r in recommendations if r['budget'] < 30_000_000]
    elif budget_filter == "Mid (\\$30M-\\$100M)":
        recommendations = [r for r in recommendations if 30_000_000 <= r['budget'] < 100_000_000]
    elif budget_filter == "Blockbuster (>\\$100M)":
        recommendations = [r for r in recommendations if r['budget'] >= 100_000_000]
    
    if genre_filter != "All Genres":
        for genre in genre_set:
            if genre_filter == genre:
                recommendations = [r for r in recommendations if genre in r['genres']]

    # Limit to requested number
    recommendations = recommendations[:n_recommendations]
    
    if not recommendations:
        st.warning("No movies found matching your filters. Expand your tastes!")
    else:
        # Display in a grid (3 columns)
        cols = st.columns(3)
        
        for i, movie in enumerate(recommendations):
            with cols[i % 3]:
                # Movie poster
                st.image(get_poster_url(movie['poster_path']), use_container_width=True)
                
                # Movie info
                st.markdown(f"**{movie['title']}**")
                
                # Similarity as progress bar
                st.progress(movie['similarity'], text=f"{movie['similarity']:.0%} match")
                
                # Details in expander (keeps UI clean)
                with st.expander("Details"):
                    st.write(f"**Rating:** {movie["vote_average"]:.1f}")
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Budget:** {format_budget(movie['budget'])}")
                    st.write(f"**Countries:** {movie['countries']}")
                    st.write(f"**Overview:** {movie['overview']}")
                    display_watch_providers(movie["id"])

else:
    st.error("Movie not found! Please try a different title.")

# FOOTER

st.divider()
st.caption(f"Database: {len(data):,} movies | Model: K-Nearest Neighbors with TF-IDF features | Thanks to TMDB (The Movie Database) for data")