from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(bytes):
    #read bytes as a pillow image
    pillow_img = Image.open(BytesIO(bytes))
    
    #convert pillow img to numpy array
    numpy_array = np.array(pillow_img)

    return numpy_array


@app.get("/")
async def ping():
    return "hello, i'm alived!"


@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    #read the file as bytes
    content = await file.read()

    #convert bytes read to numpy array
    np_img = read_file_as_image(content)

    #model.predict() function accept a batch, so we just need to add a dimension to the np.array
    img_batch = np.expand_dims(np_img, axis=0)

    #making prediction and getting confidence of that prediction
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(100*np.max(prediction), 2)

    return {
        "class": predicted_class,
        "confidence": confidence
    }


#check if this code is runned directly, it's not
#imported by other file
if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)