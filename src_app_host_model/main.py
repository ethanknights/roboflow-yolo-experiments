from fastapi import FastAPI, HTTPException, Depends
from google.cloud import storage
from ultralytics import YOLO
from typing import Dict
import os
import cv2
import logging
from inference import inference_yolo

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

app = FastAPI()
storage_client = storage.Client()

models: Dict[str, YOLO] = {}



def load_model(model_name: str):
    model_path = f"models/{model_name}.pt"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    if model_name not in models:
        models[model_name] = YOLO(model_path)
    return models[model_name]


# def download_model_from_gcs(model_name):
#     local_model_dir = 'models'
#     os.makedirs('models', exist_ok=True)
#     bucket = storage_client.bucket(f'ria-classification-models/{model_name}')
#     blob = bucket.blob(f'{model_name}.pt')
#     local_model_file_name = f'models/{model_name}.pt'
#     blob.download_to_filename(local_model_file_name)
#     return local_model_file_name


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    return image


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference/{model_name}")
async def inference(model_name: str, image_path: str):
    try:

        local_model_file_name = f'models/{model_name}.pt'
        logger.info(local_model_file_name)

        if not os.path.isfile(local_model_file_name):
            # download_model_from_gcs(model_name)
            # logger.debug(f'Successfully downloaded {model_name} to {local_model_file_name}')
            return {f'{local_model_file_name} not found and running locally only temporarily due to auth issues'}
        else:
            logger.debug(f'Successfully found {model_name} locally: {local_model_file_name}')
    except Exception as e:
        return {f"error finding file name:\n{model_name}": str(e)}

    is_external_model = 'gemini' in model_name

    if is_external_model:
        logger.debug('not setup!')
        return {"Message": "not setup!"}
    else:
        (top1_category_str,
         probabilities) = inference_yolo(local_model_file_name, image_path)

    return {
            "top1_category_str": top1_category_str,
            "probabilities_formatted": probabilities
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
