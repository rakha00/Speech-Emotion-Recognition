import streamlit as st
import torch
import torch.nn as nn
import librosa
from transformers import AutoFeatureExtractor, AutoModel
import joblib
from pathlib import Path
import numpy as np
import sys
import os

# Add the site-packages directory of the current environment to sys.path
# This helps resolve ModuleNotFoundError when packages are installed but not found
site_packages_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)

# Configuration - should match serv9.ipynb
PRETRAINED_MODEL_NAME = "openai/whisper-large-v3"
PRETRAINED_MODEL_OUTPUT_DIM = 1280
PRETRAINED_MAX_LEN_PADDING = 448
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for saved models and label encoder
MODEL_DIR = Path('./model')
# Convert Path objects to strings for compatibility with torch.load and joblib.load
BEST_MODEL_PATH = str(MODEL_DIR / f'best_lstm_with_{PRETRAINED_MODEL_NAME.split("/")[-1]}_features.pth')
LABEL_ENCODER_PATH = str(MODEL_DIR / 'label_encoder.joblib')

# Model Architecture - Re-define the LSTM model as in serv9.ipynb
class EmotionClassifierLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, num_classes: int, dropout_prob: float):
        super().__init__()
        self.lstm_layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_prob if n_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier_hidden_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu_activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        batch_size = input_features.size(0)
        num_directions = 2
        h0 = torch.zeros(self.lstm_layer.num_layers * num_directions, batch_size, self.lstm_layer.hidden_size).to(input_features.device)
        c0 = torch.zeros(self.lstm_layer.num_layers * num_directions, batch_size, self.lstm_layer.hidden_size).to(input_features.device)
        lstm_output, _ = self.lstm_layer(input_features, (h0, c0))
        last_step_output = lstm_output[:, -1, :]
        x = self.classifier_hidden_layer(last_step_output)
        x = self.relu_activation(x)
        x = self.dropout_layer(x)
        logits = self.output_layer(x)
        return logits

# Function to load models and label encoder
@st.cache_resource(hash_funcs={torch.nn.Module: lambda _: None, AutoFeatureExtractor: lambda _: None, AutoModel: lambda _: None, joblib.Parallel: lambda _: None})
def load_models():
    try:
        # Load Whisper feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)
        pretrained_model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME).to(DEVICE)
        feature_extraction_model = pretrained_model.get_encoder()
        feature_extraction_model.eval()

        # Load label encoder
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        # Load LSTM classification model
        # These parameters should match those used during training in serv9.ipynb
        # From serv9.ipynb: LSTM_HIDDEN_SIZE = 256, LSTM_NUM_LAYERS = 3, LSTM_DROPOUT = 0.3
        LSTM_HIDDEN_SIZE = 256
        LSTM_NUM_LAYERS = 3
        LSTM_DROPOUT = 0.3
        NUM_CLASSES = len(label_encoder.classes_) # Get from loaded label encoder

        classification_model = EmotionClassifierLSTM(
            input_dim=PRETRAINED_MODEL_OUTPUT_DIM,
            hidden_dim=LSTM_HIDDEN_SIZE,
            n_layers=LSTM_NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout_prob=LSTM_DROPOUT
        ).to(DEVICE)
        classification_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        classification_model.eval()

        return feature_extractor, feature_extraction_model, classification_model, label_encoder
    except Exception as e:
        st.error(f"Error loading models. Please ensure 'model' directory exists and contains the necessary files (e.g., '{BEST_MODEL_PATH}' and '{LABEL_ENCODER_PATH}'). Error: {e}")
        raise # Re-raise the exception to allow Streamlit to display it appropriately

# Prediction function - adapted from serv9.ipynb
def predict_emotion(audio_file_path: str, feature_extractor, feature_extraction_model, classification_model, label_encoder_obj) -> str:
    waveform, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, mono=True)

    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    input_data = inputs.input_features
    input_for_extraction_model = input_data.to(DEVICE)

    with torch.no_grad():
        outputs_extraction = feature_extraction_model(input_features=input_for_extraction_model)
        raw_extracted_features = outputs_extraction.last_hidden_state.squeeze(0)

    seq_len_model = raw_extracted_features.shape[0]
    final_features_for_classifier = raw_extracted_features

    if seq_len_model > PRETRAINED_MAX_LEN_PADDING:
        final_features_for_classifier = raw_extracted_features[:PRETRAINED_MAX_LEN_PADDING, :]
    elif seq_len_model < PRETRAINED_MAX_LEN_PADDING:
        padding_needed = PRETRAINED_MAX_LEN_PADDING - seq_len_model
        padding_tensor = torch.zeros(padding_needed, raw_extracted_features.shape[1], device=DEVICE)
        final_features_for_classifier = torch.cat((raw_extracted_features, padding_tensor), dim=0)

    final_features_for_classifier = final_features_for_classifier.to(DEVICE)
    feature_tensor_batch = final_features_for_classifier.unsqueeze(0)

    with torch.no_grad():
        output_logits = classification_model(feature_tensor_batch)
        _, predicted_id = torch.max(output_logits, 1)

    predicted_label_name = label_encoder_obj.inverse_transform(predicted_id.cpu().numpy())[0]
    return predicted_label_name

# Streamlit App
st.title("Speech Emotion Recognition")
st.write("Upload an audio file or record from your microphone to predict the emotion.")

# Load models once
with st.spinner("Loading models... This might take a moment."):
    feature_extractor, feature_extraction_model, classification_model, label_encoder = load_models()
st.success("Models loaded successfully!")

# Audio Upload
uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file temporarily
    temp_audio_path = Path("temp_audio.wav")
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Analyzing audio...")
    try:
        predicted_emotion = predict_emotion(
            str(temp_audio_path),
            feature_extractor,
            feature_extraction_model,
            classification_model,
            label_encoder
        )
        st.success(f"Predicted Emotion: **{predicted_emotion.upper()}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    finally:
        # Clean up temporary file
        if temp_audio_path.exists():
            temp_audio_path.unlink()

# Microphone Input
st.header("Record from Microphone")
st.warning("Microphone recording is only available when running Streamlit locally and might require additional browser permissions.")

from audio_recorder_streamlit import audio_recorder

audio_bytes = audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    temp_recorded_audio_path = Path("temp_recorded_audio.wav")
    with open(temp_recorded_audio_path, "wb") as f:
        f.write(audio_bytes)
    st.write("Analyzing recorded audio...")
    try:
        predicted_emotion = predict_emotion(
            str(temp_recorded_audio_path),
            feature_extractor,
            feature_extraction_model,
            classification_model,
            label_encoder
        )
        st.success(f"Predicted Emotion: **{predicted_emotion.upper()}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    finally:
        if temp_recorded_audio_path.exists():
            temp_recorded_audio_path.unlink()