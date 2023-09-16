import streamlit as st
import soundfile as sf
import sounddevice as sd
import os
import tempfile
import librosa
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
from keras.models import load_model

model=load_model('bestmodel.h5')
def home():
    st.subheader("Home")
    st.write("Welcome to the Speech Emotion Recognition")
    imageha = mpimg.imread('img.jpg')     
    st.image(imageha)
    st.write('')
    st.write("There are a set of 200 target words were spoken in the carrier phrase 'Say the word _'' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.")
    st.write('')
    st.write("The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format")
    col1, col2,col3 = st.columns(3)

    
    col1.header(" Happy 😊")
   


    col2.header("Sad 😭")
    
    
    col3.header("Disgust 🤢")
    
    
    col4, col5,col6 = st.columns(3)

    
    col4.header("Angry 😡")
    
    
    col5.header("Fear 😨")
    
    col6.header("Surprise 🙀")
    
# Define the prediction function
def predict_emotion(audio_data, sample_rate):
    predictions = model.predict(audio_data)
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = class_labels[predicted_emotion_index]
    return predicted_emotion
def extract_mfcc_for_prediction(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
def prediction():
    
    st.subheader("Emotion Prediction")

    # Upload audio file
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    if uploaded_audio:
        # Display audio file details
        audio_info = sf.info(uploaded_audio)
        st.audio(uploaded_audio, format="audio/wav")
        st.write(f"Audio File Details:\n"
                 f"Duration: {audio_info.duration} seconds\n"
                 f"Sample Rate: {audio_info.samplerate} Hz\n"
                 f"Channels: {audio_info.channels}")
         # Predict emotion
        if st.button("Predict Emotion"):
            st.info("Predicting emotion... Please wait.")
            # st.audio(audio_file, format='audio/wav')
            # audio_data= sf.read(uploaded_audio)
            mfcc_features = extract_mfcc_for_prediction(uploaded_audio)
            pred = np.expand_dims(mfcc_features, axis=0)
            pred = np.expand_dims(pred, axis=-1)
            emotion = predict_emotion(pred, audio_info.samplerate)
            st.success(f"Predicted Emotion: {emotion}")

       

    # Record audio
    st.subheader("Record Audio")
    recording = st.button("Start Recording")
    if recording:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
            st.info("Recording... Click 'Stop Recording' to finish.")
            audio_filename = tmp_audio.name
            audio_stream = sd.OutputStream(callback=tmp_audio.write, channels=1, dtype='int16')
            audio_stream.start()
            st.button("Stop Recording")

        if st.button("Stop Recording"):
            audio_stream.stop()
            audio_stream.close()
            st.success("Recording stopped.")
            st.audio(audio_filename, format="audio/wav")
            os.remove(audio_filename)
         # Predict emotion
        if st.button("Predict Emotion"):
            st.info("Predicting emotion... Please wait.")
            audio_data= sf.read(uploaded_audio)
            mfcc_features = extract_mfcc_for_prediction(audio_data)
            pred = np.expand_dims(mfcc_features, axis=0)
            pred = np.expand_dims(pred, axis=-1)
            emotion = predict_emotion(pred, audio_info.samplerate)
            st.success(f"Predicted Emotion: {emotion}")
    
def visualization():
    st.write("## Exploratory Data Visualization")
    col1, col2 = st.columns(2)

    original = Image.open('spec-happy.png')
    col1.header("Spectogram of Happy")
    col1.image(original, use_column_width=True)

    grayscale = Image.open('spec-sad.png')
    col2.header("Spectogram of Sad")
    col2.image(grayscale, use_column_width=True)
    
    col3, col4 = st.columns(2)

    original1 = Image.open('wave-angry.png')
    col3.header("Wave form of Angry")
    col3.image(original1, use_column_width=True)

    grayscale1 = Image.open('wave-disgust.png')
    col4.header("Wave form of Disgust")
    col4.image(grayscale1, use_column_width=True)
        



def main():
    st.set_page_config(layout="wide")
    st.title("Speech Emotion Recoginition App")
# Create the tab layout
    tabs = ["Home", "Prediction", "Exploratory Data Visualization"]
    page = st.sidebar.selectbox("Select a page", tabs)

# Show the appropriate page based on the user selection
    if page == "Home":
        home()
    elif page == "Prediction":
        prediction()
    elif page == "Exploratory Data Visualization":
        visualization()
   
main()
