import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK data if not already downloaded
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    resources = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass

# Download NLTK data
download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Emotion Classifier AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .emotion-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Define TextPreprocessor class
class TextPreprocessor:
    """Text preprocessing class"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text, use_stemming=False):
        """
        Preprocess text with following steps:
        - Lowercase conversion
        - Remove special characters and numbers
        - Tokenization
        - Remove stopwords
        - Lemmatization/Stemming
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Lemmatization or Stemming
        if use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    """Load all pre-trained models and preprocessing artifacts"""
    try:
        # Create preprocessor (don't load from pickle)
        preprocessor = TextPreprocessor()
        
        # Load LSTM model
        lstm_model = load_model('models/lstm_model.h5')
        
        # Load tokenizer
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return preprocessor, lstm_model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error(f"Please ensure all model files are in the 'models/' directory")
        return None, None, None, None

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load artifacts
preprocessor, lstm_model, tokenizer, label_encoder = load_artifacts()

# Check if models loaded successfully
models_loaded = all([preprocessor, lstm_model, tokenizer, label_encoder])

# Emotion configurations
EMOTION_CONFIG = {
    'joy': {'emoji': '😊', 'color': '#FFD700', 'bg': '#FFF9E6'},
    'sadness': {'emoji': '😢', 'color': '#4682B4', 'bg': '#E6F2FF'},
    'anger': {'emoji': '😠', 'color': '#DC143C', 'bg': '#FFE6E6'},
    'fear': {'emoji': '😨', 'color': '#8B008B', 'bg': '#F3E6FF'},
    'love': {'emoji': '❤️', 'color': '#FF1493', 'bg': '#FFE6F0'},
    'surprise': {'emoji': '😲', 'color': '#FF8C00', 'bg': '#FFF0E6'}
}

def predict_emotion(text, model_type):
    """Predict emotion from text"""
    if not models_loaded:
        return None, None, None
    
    # Preprocess text
    cleaned_text = preprocessor.preprocess(text)
    
    if model_type == "LSTM Neural Network":
        # LSTM prediction
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=100)
        prediction_prob = lstm_model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction_prob, axis=1)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        
        # All probabilities
        all_probs = {
            label_encoder.inverse_transform([i])[0]: float(prediction_prob[0][i])
            for i in range(len(prediction_prob[0]))
        }
    
    confidence = all_probs[predicted_emotion]
    
    return predicted_emotion, confidence, all_probs

# Header
st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: #667eea; margin: 0;'>😊 Emotion Classification AI</h1>
    </div>
""", unsafe_allow_html=True)

# Show error if models not loaded
if not models_loaded:
    st.error("⚠️ Models could not be loaded. Please ensure the following files exist in the 'models/' directory:")
    st.code("""
    models/
    ├── lstm_model.h5
    ├── tokenizer.pkl
    └── label_encoder.pkl
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Model Selection")
    model_type = st.radio(
        "Choose your model:",
        ["LSTM Neural Network"],
        help="LSTM Neural Network for emotion prediction"
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    show_probabilities = st.checkbox("Show All Probabilities", value=True)
    show_preprocessing = st.checkbox("Show Preprocessing Steps", value=False)
    
    st.markdown("---")
    
    st.markdown("### 📊 Supported Emotions")
    for emotion, config in EMOTION_CONFIG.items():
        st.markdown(f"{config['emoji']} **{emotion.capitalize()}**")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info(
        "This app uses Natural Language Processing (NLP) to classify text into "
        "different emotional categories using a Bidirectional LSTM neural network."
    )
    
    st.markdown("---")
    
    st.markdown("### 📈 Model Performance")
    try:
        comparison_df = pd.read_csv('models/model_comparison.csv', index_col=0)
        if 'LSTM Neural Network' in comparison_df.index:
            metrics = comparison_df.loc['LSTM Neural Network']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
                st.metric("Recall", f"{metrics['recall']:.3f}")
        else:
            st.warning("LSTM metrics not found")
    except:
        st.warning("Model metrics not available")

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analytics", "📝 History"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Enter Your Text")
        user_input = st.text_area(
            "Text Input",
            height=200,
            placeholder="Type or paste your text here...\n\nExample: I am feeling great today!",
            help="Enter any text and we'll predict the emotion",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            predict_button = st.button("🔮 Predict Emotion", type="primary")
        
        with col_btn2:
            if st.button("🧹 Clear"):
                user_input = ""
                st.rerun()
        
        # with col_btn3:
        #     if st.button("📋 Example"):
        #         user_input = "I am so excited about this amazing opportunity!"
        #         st.rerun()
    
    with col2:
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px;'>
        <p><b>For best results:</b></p>
        <ul>
            <li>Use complete sentences</li>
            <li>Be expressive with your words</li>
            <li>Include emotional context</li>
            <li>Minimum 5 words recommended</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # st.markdown("### 🎨 Example Texts")
        # examples = {
        #     "Joy": "I'm so happy and grateful for everything!",
        #     "Sadness": "I feel lonely and heartbroken today.",
        #     "Anger": "This is absolutely frustrating and unfair!",
        #     "Fear": "I'm terrified about what might happen.",
        #     "Love": "You mean everything to me, I adore you!",
        #     "Surprise": "Wow, I can't believe this happened!"
        # }
        
        # for emotion, text in examples.items():
        #     if st.button(f"{EMOTION_CONFIG[emotion.lower()]['emoji']} {emotion}", key=f"ex_{emotion}"):
        #         user_input = text
        #         st.rerun()
    
    # Prediction
    if predict_button and user_input:
        if len(user_input.strip()) < 5:
            st.warning("⚠️ Please enter at least 5 characters for better prediction accuracy.")
        else:
            with st.spinner("🤔 Analyzing emotion..."):
                predicted_emotion, confidence, all_probs = predict_emotion(user_input, model_type)
                
                if predicted_emotion:
                    # Add to history
                    st.session_state.history.append({
                        'text': user_input,
                        'emotion': predicted_emotion,
                        'confidence': confidence,
                        'model': model_type
                    })
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Results")
                    
                    # Main result
                    config = EMOTION_CONFIG[predicted_emotion]
                    
                    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                    
                    with result_col2:
                        st.markdown(f"""
                        <div style='text-align: center; background: {config['bg']}; 
                                    padding: 30px; border-radius: 15px; border: 3px solid {config['color']};'>
                            <div style='font-size: 80px;'>{config['emoji']}</div>
                            <h2 style='color: {config['color']}; margin: 10px 0;'>
                                {predicted_emotion.upper()}
                            </h2>
                            <div style='background: white; border-radius: 10px; padding: 10px; margin-top: 15px;'>
                                <p style='margin: 0; color: #666;'>Confidence Score</p>
                                <h1 style='margin: 5px 0; color: {config['color']};'>
                                    {confidence*100:.1f}%
                                </h1>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("")
                    
                    # Show preprocessing if enabled
                    if show_preprocessing:
                        with st.expander("🔍 View Preprocessing Steps"):
                            st.markdown("**Original Text:**")
                            st.code(user_input)
                            
                            st.markdown("**After Preprocessing:**")
                            cleaned = preprocessor.preprocess(user_input)
                            st.code(cleaned)
                            
                            st.markdown("**Word Count:**")
                            st.write(f"Original: {len(user_input.split())} words | "
                                   f"Cleaned: {len(cleaned.split())} words")
                    
                    # Probability distribution
                    if show_probabilities and all_probs:
                        st.markdown("### 📊 Probability Distribution")
                        
                        # Sort probabilities
                        sorted_probs = dict(sorted(all_probs.items(), 
                                                  key=lambda x: x[1], reverse=True))
                        
                        # Create bar chart
                        fig = go.Figure()
                        
                        colors = [EMOTION_CONFIG[em]['color'] for em in sorted_probs.keys()]
                        
                        fig.add_trace(go.Bar(
                            x=list(sorted_probs.keys()),
                            y=list(sorted_probs.values()),
                            marker_color=colors,
                            text=[f"{v*100:.1f}%" for v in sorted_probs.values()],
                            textposition='auto',
                            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title="Emotion Probability Scores",
                            xaxis_title="Emotion",
                            yaxis_title="Probability",
                            yaxis_tickformat='.0%',
                            height=400,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig,  width='stretch')
                        
                        # Detailed probabilities table
                        st.markdown("### 📋 Detailed Scores")
                        prob_df = pd.DataFrame({
                            'Emotion': [f"{EMOTION_CONFIG[e]['emoji']} {e.capitalize()}" 
                                      for e in sorted_probs.keys()],
                            'Probability': [f"{v*100:.2f}%" for v in sorted_probs.values()],
                            'Score': list(sorted_probs.values())
                        })
                        
                        st.dataframe(
                            prob_df[['Emotion', 'Probability']],
                            width='stretch',
                            hide_index=True
                        )
                else:
                    st.error("❌ Error making prediction. Please check if models are loaded correctly.")

with tab2:
    st.markdown("## 📊 Model Performance Analytics")
    
    try:
        comparison_df = pd.read_csv('models/model_comparison.csv', index_col=0)
        
        # Metrics comparison
        st.markdown("### 🏆 Model Comparison")
        
        fig = go.Figure()
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=comparison_df.index,
                y=comparison_df[metric],
                text=[f"{v:.3f}" for v in comparison_df[metric]],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Performance Metrics Across All Models",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig,  width='stretch')
        
        # Best model highlight
        best_model = comparison_df['accuracy'].idxmax()
        best_acc = comparison_df['accuracy'].max()
        
        st.success(f"🏆 **Best Model:** {best_model} with {best_acc:.2%} accuracy")
        
        # Detailed table
        st.markdown("### 📋 Detailed Performance Table")
        
        styled_df = comparison_df.style.format({
            'accuracy': '{:.4f}',
            'f1_score': '{:.4f}',
            'precision': '{:.4f}',
            'recall': '{:.4f}'
        }).background_gradient(cmap='RdYlGn', subset=metrics)
        
        st.dataframe(styled_df, width='stretch')
        
    except Exception as e:
        st.warning("Model comparison data not available")
        st.info("Train models using the Jupyter notebook to generate comparison data.")

with tab3:
    st.markdown("## 📝 Prediction History")
    
    if st.session_state.history:
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(st.session_state.history))
        
        with col2:
            emotions = [h['emotion'] for h in st.session_state.history]
            most_common = Counter(emotions).most_common(1)[0]
            st.metric("Most Common", 
                     f"{EMOTION_CONFIG[most_common[0]]['emoji']} {most_common[0]}")
        
        with col3:
            avg_conf = np.mean([h['confidence'] for h in st.session_state.history])
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        
        st.markdown("---")
        
        # Emotion distribution pie chart
        emotion_counts = Counter(emotions)
        
        fig = px.pie(
            values=list(emotion_counts.values()),
            names=[f"{EMOTION_CONFIG[e]['emoji']} {e.capitalize()}" 
                   for e in emotion_counts.keys()],
            title="Emotion Distribution in History",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig,  width='stretch')
        
        # History table
        st.markdown("### 📜 Recent Predictions")
        
        history_df = pd.DataFrame(st.session_state.history[::-1])  # Reverse for recent first
        history_df['emotion_display'] = history_df['emotion'].apply(
            lambda x: f"{EMOTION_CONFIG[x]['emoji']} {x.capitalize()}"
        )
        history_df['confidence_display'] = history_df['confidence'].apply(
            lambda x: f"{x*100:.1f}%"
        )
        
        display_df = history_df[['text', 'emotion_display', 'confidence_display', 'model']].head(10)
        display_df.columns = ['Text', 'Emotion', 'Confidence', 'Model']
        
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the Predict tab to get started!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 10px;'>
        <p style='margin: 5px 0 0 0; color: #999; font-size: 12px;'>
            © 2024 Emotion Classification AI | All Rights Reserved
        </p>
    </div>
""", unsafe_allow_html=True)