from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "allan-tf-models"
class_names = ["Early Blight", "Late Blight", "Healthy"]

model = None

#blob => binary large object
#this function 'll be running on a different server on gc and that server
#will download the model from the bucket, it's destination_file_name
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    #need to execute this function only on the first call
    global model 
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/model.h5",
            "/tmp/model.h5"
        )

        model = tf.keras.models.load_model("/tmp/model.h5")

    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    image = image/255

    #predict function expects a batch
    img_batch = tf.expand_dims(image, 0)
    
    prediction = model.predict(img_batch)
    
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': confidence
    }