import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Function to generate captions
def generate_captions(image, object_choice=None, prefix1="", prefix2="", prefix3=""):
    captions = []
    for prefix in [prefix1, prefix2, prefix3]:
        if prefix and object_choice and object_choice != "None":
            caption_prefix = f"{prefix} {object_choice} in the image is"
            inputs = processor(image, caption_prefix, return_tensors="pt")
        else:
            inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

# Streamlit interface
st.image("logo.png", use_column_width=False)
st.title("Image-Based Object Condition Assessment")
st.write("Upload an image and optionally select or type an object to include in the caption.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Dropdown for selecting the object and allowing custom input
    object_choice = st.selectbox(
        "Choose or type an object:",
        ("None", "carpet", "sofa", "table", "chair", "lamp", "glass", "wall", "bed", "other"),
        index=0  # Default to "None"
    )

    if object_choice == "other":
        object_choice = st.text_input("Type your object:")

    # Input fields for three prefix statements
    prefix1 = st.text_input("Enter the first prefix statement:", "The defect of the")
    prefix2 = st.text_input("Enter the second prefix statement:", "The condition of the")
    prefix3 = st.text_input("Enter the third prefix statement:", "Inspecting the")

    # Button to generate captions
    if st.button("Generate Captions"):
        captions = generate_captions(image, object_choice, prefix1, prefix2, prefix3)
        for i, caption in enumerate(captions, 1):
            st.write(f"Caption {i}: {caption}")
