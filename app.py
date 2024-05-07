import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Function to generate captions
def generate_caption(image, include_object, object_choice=None):
    if include_object and object_choice:
        caption_prefix = f"The {object_choice} in the image is"
        inputs = processor(image, caption_prefix, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit interface
st.title("Image Captioning Application")
st.write("Upload an image and optionally select an object to include in the caption.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Checkbox to choose whether to include an object in the caption
    include_object = st.checkbox("Include an object in the caption?")

    object_choice = None
    if include_object:
        # Dropdown for selecting the object if the user chooses to include it
        object_choice = st.selectbox(
            "Choose an object:",
            ("carpet", "sofa", "table", "chair", "lamp"),
            index=0  # Default to the first item
        )

    # Button to generate caption
    if st.button("Generate Caption"):
        caption = generate_caption(image, include_object, object_choice)
        st.write(caption)

