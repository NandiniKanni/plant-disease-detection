# from fastapi import FastAPI,File,UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# from pathlib import Path


# app=FastAPI()
# app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



# MODEL = tf.keras.models.load_model("potato_disease_model.h5")
# CLASS_NAMES=["Early Blight","Late Blight","Healthy"]

# @app.get("/")
# async def serve_frontend():
#     return HTMLResponse(Path("frontend/index.html").read_text())

# @app.get("/ping")
# async def ping():
#     return "hello!i am here"

# # def read_file_as_image(data)->np.ndarray:
# #     image=np.array(Image.open(BytesIO(data)))
# #     return 


# # @app.post("/predict")
# # async def predict(
# #      file:UploadFile=File(...)
# # ):
# #     #bytes=await file.read()
# #     image=read_file_as_image(await file.read())
# #     return image  
# #     # pass
# #    # return {"message": "API working"}

# def read_file_as_image(data) -> np.ndarray:
#     return np.array(Image.open(BytesIO(data)))

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     image_batch = np.expand_dims(image, axis=0)

#     prediction = MODEL.predict(image_batch)
#     confidence=float(np.max(prediction[0]))
#      # 🔥 If confidence is too low, reject it
#     if confidence < 0.70:
#         return {
#             "predicted_class": "Unknown",
#             "confidence": confidence,
#             "message": "Not a recognizable potato leaf. Please upload a clear potato leaf image."
#         }
#     predicted_class= CLASS_NAMES[np.argmax(prediction[0])]
    
#     return {"prediction": prediction.tolist(),
#             'predicted_class':predicted_class,
#             "confidence":confidence
#             }
#    # pass

#     #return {"shape": image.shape}

# if __name__=="__main__":
#     uvicorn.run(app,host='localhost',port=8000)
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("potato_disease_model.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def serve_frontend():
    return HTMLResponse(Path("frontend/index.html").read_text(encoding="utf-8"))

@app.get("/ping")
async def ping():
    return "hello! i am here"

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    img = img.convert("RGB")        # handles PNG, RGBA, grayscale etc.
    img = img.resize((256, 256))    # match training size
    return np.array(img)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()    # read ONCE
        image = read_file_as_image(data)
        image_batch = np.expand_dims(image, axis=0)

        prediction = MODEL.predict(image_batch)
        confidence = float(np.max(prediction[0]))

        if confidence < 0.70:
            return {
                "predicted_class": "Unknown",
                "confidence": confidence,
                "message": "Not a recognizable potato leaf."
            }

        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "message": "ok"
        }

    except Exception as e:
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)