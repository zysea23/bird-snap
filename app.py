import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import json
from PIL import Image
import io
import base64
import wikipedia
from streamlit_lottie import st_lottie


# streamlit run app.py  # run this for local running

st.set_page_config(page_title="Bird Snap") 

# Hide hamburger menu and Streamlit watermark
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}

            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ---------------------------------- Functions ----------------------------------
@st.cache_data
def load_index_to_class(json_file_name: str = "index_to_class.json") -> dict[int, str]:
    """
    Loads a JSON file containing the index to class name mapping and returns it as a dictionary.

    Args:
        json_file_name (str): The path to the JSON file.

    Returns:
        dict: The index to class name mapping as a dictionary.
    """
    with open(f"models and data/{json_file_name}", "r") as json_file:
        index_to_class_dict = json.load(json_file)
    return index_to_class_dict


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "bird_10.keras") -> tf.keras.Model:
    """
    Loads the specified TensorFlow model.

    Args:
        model_name (str): The name of the model file.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    tf_model = tf.keras.models.load_model(f"models and data/{model_name}")
    return tf_model


def display_bird_summary(best_guess_row: pd.Series) -> None:
    """
    Displays the bird summary, including the Wikipedia description and an image.

    Args:
    best_guess_row (pd.Series): A pandas Series representing the best guess for the bird species.
                                It should contain the column "Common Name".
    """
    wiki_description = get_bird_description(best_guess_row["Common Name"])
    if wiki_description is not None:  # only is None if can't find species, so we shouldn't do anything
        other_image = Image.open(f"models and data/sample photos/{best_guess_row['Common Name']}.jpg")

        image_width = 300
        image_bytes = io.BytesIO()
        other_image.save(image_bytes, format="JPEG")
        image_html = f'<img src="data:image/jpeg;base64,{base64.b64encode(image_bytes.getvalue()).decode()}" ' \
                     f'alt="Bird Image" style="float: right; width: {image_width}px; margin-left: 20px;">'

        st.markdown(f'{image_html} {wiki_description}', unsafe_allow_html=True)

        st.info(f'Read more about the [{best_guess_row["Common Name"].replace("_", " ")} on Wikipedia](https://en.wikipedia.org/wiki/{best_guess_row["Common Name"]})')


@st.cache_resource
def prep_image(img: bytes, shape: int = 300, scale: bool = False) -> tf.Tensor:
    """
    Preprocesses the image data.

    Args:
        img (bytes): The image data as a byte string.
        shape (int): The desired shape for the image (default: 260).
        scale (bool): Whether to scale the pixel values (default: False).

    Returns:
        The preprocessed image as a TensorFlow tensor.
    """
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=([shape, shape]))
    if scale:
        img = img/255.
    return img


def classify_image(img: bytes, model: tf.keras.Model) -> pd.DataFrame:
    """
    Classifies the given image using the provided model and returns a DataFrame
    containing the top 3 predictions and their probabilities.

    Args:
        img (bytes): The image to be classified.
        model (tf.keras.Model): The pre-trained model to use for prediction.

    Returns:
        A pandas DataFrame containing the top 3 predictions and their probabilities,
        sorted in descending order of probability.
    """
    # Preprocess the image
    img = prep_image(img)
    # Expand dimensions to create a batch of size 1
    img = tf.cast(tf.expand_dims(img, axis=0), tf.float32)

    # Make predictions using the model
    pred_probs = model.predict(img)

    # Get the indices of the top 3 predicted classes
    top_3_indices = pred_probs.argsort()[0][-3:][::-1]  #  sorted？

    # Compute the probabilities for the top 3 predictions
    values = pred_probs[0][top_3_indices] * 100
    labels = [index_to_class[str(i)] for i in top_3_indices]

    # Create a DataFrame to store the top 3 predictions and their probabilities
    prediction_df = pd.DataFrame({
        "Common Name": labels,
        "Probability": values,
    })

    # Sort the DataFrame by Probability
    return prediction_df.sort_values("Probability", ascending=False)


@st.cache_data
def get_bird_description(bird_name: str) -> str:
    """
    Fetches a short description of the bird from Wikipedia.

    Args:
        bird_name (str): The common name of the bird.

    Returns:
        str: The summary from Wikipedia if available, otherwise None.
    """
    try:
        return wikipedia.page(bird_name.replace("_", " ")).summary
    except wikipedia.exceptions.PageError:
        return None


# Load index-to-class mapping and model
index_to_class = load_index_to_class()

# ---------------------------------- SideBar ----------------------------------

st.sidebar.title('🐦Welcome to Bird Snap')
st.sidebar.write('''
                 
🦜 Bird Snap – Your AI Bird-Watching Buddy!
Ever wondered what bird just fluttered by? 🧐 Snap a picture, and Bird Snap will tell you instantly! With AI-powered bird recognition, you’ll never be left guessing again.

🌿 What’s so cool about it?

Identify 500+ bird species just by taking a photo! 📸
                 
Trained on 50,000+ stunning bird images, so no feather goes unnoticed. 🪶
                 
Powered by cutting-edge AI magic (a.k.a. transfer learning), making it smarter with every snap.

🌟 Get ready to become a birding legend!

From backyard visitors to rare finds in the wild, Bird Snap is your personal bird expert—right in your pocket! 🔍✨
''')
st.sidebar.markdown(body="""

""", unsafe_allow_html=True)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
lottie_coding = load_lottiefile("Animation.json")
st_lottie(
    lottie_coding,
    key="bird_animation",
    height=280,  
    width=280,   
    loop=True    
)

st.sidebar.write(
    """
    <div style="text-align: center; margin-top: 50px; padding-bottom: 50px;">
        <a href="https://github.com/zysea23/Bird-Snap"><strong>Learn more</strong></a> about this website and the underlying machine learning model.
        
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------- Main Body ----------------------------------

st.title("See a Bird? Snap It, Know It! 🐤🔍")
st.header("Identify what kind of bird you snapped a photo of!")

file = st.file_uploader(label="Upload an image of a bird, let us name it for you!",
                        type=["jpg", "jpeg", "png"])

if not file:
    image = None  # set to None because it doesn't exist yet
    pred_button = st.button("Identify Species", disabled=True, help="Upload an image to make a prediction")
    st.stop()
else:
    image = file.read()
    # Center the image using st.columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Display image in center column
        st.image(image, use_container_width=True)
    pred_button = st.button("Identify Species")

if pred_button:
    # Perform image classification and obtain prediction, confidence, and DataFrame
    with st.spinner("Loading Image Classification Model..."):
        model = load_model()

    with st.spinner("Classifying Image..."):
        df = classify_image(image, model)

    top_prediction_row = df.iloc[0]
    # Display the prediction and confidence
    st.success(f'Predicted Species: **{top_prediction_row["Common Name"].replace("_", " ")}** Confidence: {top_prediction_row["Probability"]:.2f}%')

    # create list for y-axis of plot which displays the name of the bird and links to the wikipedia page
    y_axis_wiki_namelist = []
    for index, row in df.iterrows():
        label = row["Common Name"]
        current_link = f'https://en.wikipedia.org/wiki/{label}'
        y_axis_wiki_namelist.append(f'<a href="{current_link}" target="_blank">{label.replace("_", " ")}</a>')

    fig = go.Figure(data=[
        go.Bar(
            x=df["Probability"],
            y=y_axis_wiki_namelist,
            orientation="h",
            text=df["Probability"].apply(lambda x: f"{x:.2f}%"),
            textposition="auto",
            marker=dict(color="plum"),  # need to decide on a good colour
        )
    ])

    fig.update_layout(
        title="Common Name",
        xaxis_title="Probability",
        yaxis_title="Species",
        width=600,
        height=400,
        dragmode=False,
    )

    st.plotly_chart(fig)

    display_bird_summary(best_guess_row=top_prediction_row)
