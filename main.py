from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
from fastapi.middleware.cors import CORSMiddleware


# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI() 

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("./model_vgg19.h5")

def model_predict(img, model):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # Resize the image to the required input shape
    img = img.resize((224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # Scaling
    #x=x/255
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    preds=np.argmax(pred, axis=1)
    if preds==0:
        preds="The Person is Infected With Malaria"
    else:
        preds="The Person is not Infected With Malaria"
    
    return preds

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    img = read_file_as_image(await file.read())
    preds = model_predict(img, model)
    result= preds
    return result

if __name__ == "__main__": 
    uvicorn.run(app, host='localhost', port=8000)