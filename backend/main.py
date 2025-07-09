
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Configure logging
logging.basicConfig(filename='image_classification.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logging.info("Received image upload request.")
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logging.info("Image successfully loaded.")

        # Preprocess image
        processed_image = preprocess_image(image)
        logging.info("Image successfully resized and preprocessed.")

        # Make prediction
        predictions = model.predict(processed_image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        logging.info(f"Prediction successful: {decoded_predictions}")

        # Format response
        response = [{"label": label, "probability": float(prob)} for (_, label, prob) in decoded_predictions]
        logging.info(f"Sending response: {response}")
        return response

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Image Classification API"}
