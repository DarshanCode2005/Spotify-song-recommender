import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import torch

# Initialize emotion classifier (do this once at startup)
@st.cache_resource
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/songs.csv')
    return df

def preprocess_features(df):
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'valence', 'tempo', 'loudness']
    
    X = df[features].copy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=features), features

def get_mood_features(emotion):
    """
    Map emotions to music features with more nuanced mappings
    """
    mood_features = {
        'joy': {
            'valence': 0.9,
            'energy': 0.8,
            'danceability': 0.8,
            'tempo': 0.7,
            'acousticness': 0.3
        },
        'sadness': {
            'valence': 0.2,
            'energy': 0.3,
            'danceability': 0.3,
            'tempo': 0.4,
            'acousticness': 0.7
        },
        'anger': {
            'valence': 0.3,
            'energy': 0.9,
            'danceability': 0.6,
            'tempo': 0.8,
            'acousticness': 0.2
        },
        'fear': {
            'valence': 0.2,
            'energy': 0.7,
            'danceability': 0.4,
            'tempo': 0.6,
            'acousticness': 0.4
        },
        'love': {
            'valence': 0.8,
            'energy': 0.5,
            'danceability': 0.6,
            'tempo': 0.5,
            'acousticness': 0.6
        },
        'surprise': {
            'valence': 0.7,
            'energy': 0.7,
            'danceability': 0.7,
            'tempo': 0.6,
            'acousticness': 0.4
        }
    }
    return mood_features.get(emotion, mood_features['joy'])

def get_recommendations(df, mood_features, n_recommendations=5):
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'valence', 'tempo', 'loudness']
    
    X, _ = preprocess_features(df)
    
    # Create mood vector with weighted features
    mood_vector = np.zeros(len(features))
    for i, feature in enumerate(features):
        if feature in mood_features:
            mood_vector[i] = mood_features[feature]
        else:
            mood_vector[i] = 0.5
    
    # Calculate weighted Euclidean distances
    distances = np.linalg.norm(X - mood_vector, axis=1)
    
    # Get top N recommendations
    recommended_indices = distances.argsort()[:n_recommendations]
    recommendations = df.iloc[recommended_indices]
    
    return recommendations[['artist_name', 'track_name', 'genre']]

def analyze_mood(text, classifier):
    """
    Analyze text and return emotion label and confidence score
    """
    # Get emotion scores
    scores = classifier(text)[0]
    
    # Get top emotion and its score
    top_emotion = max(scores, key=lambda x: x['score'])
    
    return top_emotion['label'], top_emotion['score']

def get_emotion_emoji(emotion):
    """
    Return appropriate emoji for each emotion
    """
    emoji_map = {
        'joy': '😊',
        'sadness': '😢',
        'anger': '😠',
        'fear': '😨',
        'love': '❤️',
        'surprise': '😮'
    }
    return emoji_map.get(emotion, '🎵')

# Streamlit UI
st.title("🎵 Enhanced Mood-Based Song Recommender")
st.write("Chat with me and I'll recommend songs based on your emotional state!")

# Initialize chat history and emotion classifier
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load data and classifier
df = load_data()
emotion_classifier = load_emotion_classifier()

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

    # Analyze emotion and get recommendations
    emotion, confidence = analyze_mood(prompt, emotion_classifier)
    mood_features = get_mood_features(emotion)
    recommendations = get_recommendations(df, mood_features)

    # Create detailed response
    emoji = get_emotion_emoji(emotion)
    response = f"I sense that you're feeling **{emotion}** {emoji} (confidence: {confidence:.2%}). Here are some songs that might resonate with your current emotional state:\n\n"
    
    for _, row in recommendations.iterrows():
        response += f"- {row['track_name']} by {row['artist_name']} ({row['genre']})\n"
    
    # Add explanation of the recommendation
    response += f"\n\nI selected these songs because they match the musical characteristics typically associated with {emotion}:"
    if emotion in ['joy', 'love']:
        response += "\n- Uplifting melodies with higher valence"
        response += "\n- Moderate to high energy"
        response += "\n- Good danceability"
    elif emotion in ['sadness']:
        response += "\n- More acoustic elements"
        response += "\n- Lower tempo and energy"
        response += "\n- More introspective qualities"
    elif emotion in ['anger']:
        response += "\n- High energy tracks"
        response += "\n- Strong rhythmic elements"
        response += "\n- Intense musical characteristics"
    elif emotion in ['fear']:
        response += "\n- Atmospheric and tense elements"
        response += "\n- Moderate to high energy"
        response += "\n- Complex musical patterns"
    elif emotion in ['surprise']:
        response += "\n- Dynamic and varied elements"
        response += "\n- Balanced energy and valence"
        response += "\n- Interesting musical progressions"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Add mood analysis explanation in the sidebar
with st.sidebar:
    st.title("About the Emotion Analysis")
    st.write("""
    This app uses a state-of-the-art emotion detection model (DistilBERT) to understand your emotional state.
    It can detect six primary emotions:
    
    - Joy 😊
    - Sadness 😢
    - Anger 😠
    - Fear 😨
    - Love ❤️
    - Surprise 😮
    
    The model analyzes your message and matches songs based on audio features that correspond to each emotion.
    """) 