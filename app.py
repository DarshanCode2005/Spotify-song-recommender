import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/songs.csv')
    return df

def preprocess_features(df):
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'valence', 'tempo', 'loudness']
    
    # Create a copy of selected features
    X = df[features].copy()
    
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=features), features

def get_mood_features(mood):
    # Define mood to music feature mapping
    mood_features = {
        'happy': {'valence': 0.8, 'energy': 0.8, 'danceability': 0.8},
        'sad': {'valence': 0.2, 'energy': 0.3, 'danceability': 0.3},
        'angry': {'valence': 0.3, 'energy': 0.9, 'danceability': 0.5},
        'relaxed': {'valence': 0.6, 'energy': 0.3, 'danceability': 0.4},
        'excited': {'valence': 0.9, 'energy': 0.9, 'danceability': 0.9}
    }
    return mood_features.get(mood, mood_features['happy'])

def get_recommendations(df, mood_features, n_recommendations=5):
    # Create feature vector for mood
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'valence', 'tempo', 'loudness']
    
    # Calculate distances between mood features and songs
    X, _ = preprocess_features(df)
    
    # Create mood vector (simplified version)
    mood_vector = np.zeros(len(features))
    for i, feature in enumerate(features):
        if feature in mood_features:
            mood_vector[i] = mood_features[feature]
        else:
            mood_vector[i] = 0.5  # Default value for undefined features
    
    # Calculate Euclidean distances
    distances = np.linalg.norm(X - mood_vector, axis=1)
    
    # Get top N recommendations
    recommended_indices = distances.argsort()[:n_recommendations]
    recommendations = df.iloc[recommended_indices]
    
    return recommendations[['artist_name', 'track_name', 'genre']]

def analyze_mood(text):
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    # Map sentiment score to mood
    if compound_score >= 0.5:
        if compound_score >= 0.75:
            return 'excited'
        return 'happy'
    elif compound_score <= -0.5:
        if compound_score <= -0.75:
            return 'sad'
        return 'angry'
    else:
        return 'relaxed'

# Streamlit UI
st.title("🎵 Mood-Based Song Recommender")
st.write("Chat with me and I'll recommend songs based on your mood!")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load data
df = load_data()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Analyze mood and get recommendations
    mood = analyze_mood(prompt)
    mood_features = get_mood_features(mood)
    recommendations = get_recommendations(df, mood_features)

    # Add assistant's response to chat history
    response = f"Based on our conversation, I sense that you're feeling **{mood}**. Here are some songs that might match your mood:\n\n"
    for _, row in recommendations.iterrows():
        response += f"- {row['track_name']} by {row['artist_name']} ({row['genre']})\n"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response) 