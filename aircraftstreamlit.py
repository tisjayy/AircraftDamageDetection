import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load saved model once when app starts

def build_model():
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    output = tf.keras.layers.Flatten()(base_model.output)
    base_model = tf.keras.Model(base_model.input, output)

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights("aircraft_model.weights.h5")
    return model

@st.cache_resource
def load_model():
    return build_model()


model = load_model()

# Load BLIP processor and model once
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, blip_model

processor, blip_model = load_blip()

def preprocess_img(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_damage(img_array):
    pred = model.predict(img_array)[0][0]
    if pred < 0.5:
        return "crack", 1 - pred  # confidence in dent class
    else:
        return "dent", pred  # confidence in crack class


def generate_caption(img: Image.Image):
    inputs = processor(images=img, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

st.title("Aircraft Damage Detection and Captioning")

uploaded_file = st.file_uploader("Upload an aircraft damage image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = preprocess_img(img)
    
    label, confidence = predict_damage(img_array)
    st.write(f"**Prediction:** {label} (confidence: {confidence:.2f})")
    
    with st.spinner("Generating caption..."):
        caption = generate_caption(img)
    st.write(f"**Caption:** {caption}")
