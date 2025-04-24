# import streamlit as st
# import requests
# import numpy as np
# import cv2
# from PIL import Image

# st.title("Handwriting Recognition Keyboard")

# # Drawing Canvas
# canvas_result = st.file_uploader("Upload Handwritten Image")

# if canvas_result is not None:
#     img = Image.open(canvas_result).convert("L")  # Convert to grayscale
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # Convert image to bytes and send to Flask backend
#     img_bytes = np.array(img)
#     _, img_encoded = cv2.imencode(".png", img_bytes)

#     response = requests.post(
#         "http://127.0.0.1:5000/predict", 
#         files={"image": img_encoded.tobytes()}
#     )

#     if response.status_code == 200:
#         result = response.json()["text"]
#         st.write("**Recognized Text:**", result)
#     else:
#         st.write("Error in processing.")



import streamlit as st
import requests
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Title
st.title("Handwriting Recognition")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=5,
    stroke_color="black",
    background_color="white",
    height=150,
    width=500,
    drawing_mode="freedraw",
    key="canvas",
)

# When user submits
from io import BytesIO

if st.button("Predict"):
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))
        image = image.convert("L")
        image = image.resize((128, 32))
        image = np.array(image)

        _, img_encoded = cv2.imencode(".png", image)
        img_bytes = BytesIO(img_encoded.tobytes())
        files = {"image": ("image.png", img_bytes, "image/png")}

        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        st.write("Prediction:", response.json().get("text", "No text detected"))
