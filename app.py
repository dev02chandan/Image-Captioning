import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Load the BLIP model and processor
model_name = "dev02chandan/maitri-finetuned-blip"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Function to generate captions
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit interface
st.image("logo.png", use_column_width=False)
st.title("Image-Based Object Condition Assessment")
st.write("Upload an image to generate a caption.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    caption = generate_caption(image)
    st.write(f"Generated Caption: {caption}")