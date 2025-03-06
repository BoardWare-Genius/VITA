import uvicorn
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request, Response, status, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional

import argparse

from contextlib import asynccontextmanager
import torch
import os 

import pathlib
import hashlib
# class FormData(BaseModel):
#     prompt: str
#     vision: File | None = None
#     audio: File | None = None

def is_image(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions
@asynccontextmanager
async def lifespan(app: FastAPI):
    load()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

history = []
from PIL import Image
@app.post("/chat_completions")
async def chat_completions(request: Request):
    data = await request.form()
    query = data.get("query")
    vis_file = data.get("vis_file")
    audio_file = data.get("audio_file")
    reset = data.get("reset")
    if reset == "true":
        await _reset()

    ####### save files including images/vedios or audios ############
    path = pathlib.Path('userdata')
    if not path.exists():
        path.mkdir()
	
    # for file in files:# 迭代上传的文件
    file = vis_file
    res = await file.read()#读取文件完成后，再进行后面的操作
    filemd5 = hashlib.md5(res).hexdigest()+'.jpg'#计算md5并返回字符串
    with open(path.joinpath(filemd5), "wb") as f:
        f.write(res)#写入文件，文件名即md5值
        print(path.joinpath(filemd5))
    # filename=file.filename #文件原本的文件名
    # filetype=filename.split('.')[-1]#文件名后缀
    filepath=path.joinpath(filemd5)
    print("The md5 path is image or not {}".format(is_image(filepath)))


    print(query, vis_file.filename, audio_file)
    img = Image.open(filepath)
    print(img)
    history.append(query)
    print(history)
    return JSONResponse(content={"message": str(history)},status_code=200)
    # return {"message": history}
async def fake_video_streamer():
    for i in range(10):
        yield str(i)


@app.get("/")
async def main():
    return StreamingResponse(fake_video_streamer())

# @app.get("/")
# def main():
#     def iterfile():  # (1)
#         with open(some_file_path, mode="rb") as file_like:  # (2)
#             yield from file_like  # (3)

#     return StreamingResponse(iterfile(), media_type="video/mp4")

@app.post("/reset")
async def _reset():
    global history
    history = []
    print(history)
    path = pathlib.Path('userdata')
    for file in path.iterdir():
        file.unlink()
    return {"message": "Success"}

def load():
    global d
    d = 'dddd'
def serve():
    uvicorn.run(app="test:app", host="0.0.0.0", port=8000,reload=True)

if __name__ == "__main__":
    serve()
    # uvicorn.run(app="test:app", host="0.0.0.0", port=8000,reload=True)