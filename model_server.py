# # from flask import Flask, request, jsonify
# # import numpy as np
# # import cv2
# # '''from tensorflow.keras.models import load_model'''

# # app = Flask(__name__)

# # # Load trained handwriting recognition model
# # '''model = load_model("handwriting_model.h5")'''

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     file = request.files["image"]
# #     image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
# #     image = cv2.resize(image, (128, 32)) / 255.0  # Normalize
# #     image = np.expand_dims(image, axis=[0, -1])  # Model input shape

# # '''
# #     prediction = model.predict(image)
# #     recognized_text = decode_ctc(prediction)  # Custom function for CTC decoding

# #     return jsonify({"text": recognized_text})'
# # '''

# # if __name__ == "__main__":
# #     app.run(debug=True)




# from flask import Flask, request, jsonify
# import numpy as np
# import cv2

# app = Flask(__name__)

# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files.get("image")
#     if file is None:
#         return jsonify({"error": "No file uploaded"}), 400

#     # Read and preprocess image
#     image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (128, 32)) / 255.0  # Normalize
#     image = np.expand_dims(image, axis=[0, -1])  # Model input shape

#     # Since there's no model, return a dummy response
#     return jsonify({"message": "Image received and processed (No model available)"}), 200

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import cv2
import io

app = Flask(__name__)

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No file uploaded"}), 400

    # Read image file and convert it to a PIL Image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Preprocess image for TrOCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values


    # Generate prediction
    generated_ids = model.generate(pixel_values)
    recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"text": recognized_text})

if __name__ == "__main__":
    app.run(debug=True)
