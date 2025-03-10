import torch
import os
import argparse
import numpy as np
import copy
import gradio as gr
import re
import torchaudio
import io
import cv2
from vita.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_IMAGE_TOKEN, MAX_IMAGE_LENGTH, MIN_IMAGE_LENGTH
from vita.conversation import conv_templates, SeparatorStyle
from vita.util.mm_utils import tokenizer_image_token, tokenizer_image_audio_token 
from vita.model.builder import load_pretrained_model
from vita.model.vita_tts.decoder.llm2tts import llm2TTS
from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image
from decord import VideoReader, cpu
from vllm import LLM, SamplingParams
decoder_topk = 2
codec_chunk_size = 40
codec_padding_size = 10

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
import time
import wave
import soundfile
from scipy.io import wavfile 

import pathlib
import hashlib

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    parser = argparse.ArgumentParser(description='Run the web demo with your model path.')
    parser.add_argument('--model_path', type=str, help='Path to the model',default='./demo_VITA_ckpt/')
    parser.add_argument('--empty', type=bool, help='Whether to clean the metadata saved',default=False)
    args = parser.parse_args()
    print(args)
    load_model(args.model_path)
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if args.empty:
        import shutil
        shutil.remove('userdata')
        os.mkdir('userdata')

app = FastAPI(lifespan=lifespan)


PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

import math
from numba import jit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@jit
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def remove_special_characters(input_str):
    # Remove special tokens
    special_tokens = ['☞', '☟', '☜', '<unk>', '<|im_end|>']
    for token in special_tokens:
        input_str = input_str.replace(token, '')
    return input_str


def replace_equation(sentence):
    special_notations = {
        "sin": " sine ",
        "cos": " cosine ",
        "tan": " tangent ",
        "cot": " cotangent ",
        "sec": " secant ",
        "csc": " cosecant ",
        "log": " logarithm ",
        "exp": "e^",
        "sqrt": "根号 ",
        "abs": "绝对值 ",
    }
    
    special_operators = {
        "+": "加",
        "-": "减",
        "*": "乘",
        "/": "除",
        "=": "等于",
        '!=': '不等于',
        '>': '大于',
        '<': '小于',
        '>=': '大于等于',
        '<=': '小于等于',
    }

    greek_letters = {
        "α": "alpha ",
        "β": "beta ",
        "γ": "gamma ",
        "δ": "delta ",
        "ε": "epsilon ",
        "ζ": "zeta ",
        "η": "eta ",
        "θ": "theta ",
        "ι": "iota ",
        "κ": "kappa ",
        "λ": "lambda ",
        "μ": "mu ",
        "ν": "nu ",
        "ξ": "xi ",
        "ο": "omicron ",
        "π": "派 ",
        "ρ": "rho ",
        "σ": "sigma ",
        "τ": "tau ",
        "υ": "upsilon ",
        "φ": "phi ",
        "χ": "chi ",
        "ψ": "psi ",
        "ω": "omega "
    }

    sentence = sentence.replace('**', ' ')

    sentence = re.sub(r'(?<![\d)])-(\d+)', r'负\1', sentence)

    for key in special_notations:
        sentence = sentence.replace(key, special_notations[key]) 
    for key in special_operators:
        sentence = sentence.replace(key, special_operators[key])
    for key in greek_letters:
        sentence = sentence.replace(key, greek_letters[key])


    sentence = re.sub(r'\(?(\d+)\)?\((\d+)\)', r'\1乘\2', sentence)
    sentence = re.sub(r'\(?(\w+)\)?\^\(?(\w+)\)?', r'\1的\2次方', sentence)
    
    return sentence


def is_video(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def is_image(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

def is_wav(file_path):
    wav_extensions = {'.wav'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in wav_extensions

def load_model_embemding(model_path):
    config_path = os.path.join(model_path, 'origin_config.json')
    config = VITAQwen2Config.from_pretrained(config_path)
    model = VITAQwen2ForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True)
    embedding = model.get_input_embeddings()
    del model
    return embedding

def split_into_sentences(text):
    sentence_endings = re.compile(r'[，。？\n！？、,?.!]')
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def convert_webm_to_mp4(input_file, output_file):
    try:
        cap = cv2.VideoCapture(input_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
    except Exception as e:
        print(f"Error: {e}")
        raise


def _get_rawvideo_dec(video_path, max_frames=MAX_IMAGE_LENGTH, min_frames=MIN_IMAGE_LENGTH, video_framerate=1, s=None, e=None):
    if s is None or e is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = max(start_time, 0)
        end_time = max(end_time, 0)
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1

    if num_frames > 0:
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))
        all_pos = list(range(f_start, f_end + 1, t_stride))

        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        elif len(all_pos) < min_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f).convert("RGB") for f in vreader.get_batch(sample_pos).asnumpy()]
        return patch_images, len(patch_images)
    else:
        print(f"video path: {video_path} error.")

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0

    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0 and count % 2 == 1:
                line = line.replace("`", r"\`")
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                line = line.replace(" ", "&nbsp;")
                line = line.replace("*", "&ast;")
                line = line.replace("_", "&lowbar;")
                line = line.replace("-", "&#45;")
                line = line.replace(".", "&#46;")
                line = line.replace("!", "&#33;")
                line = line.replace("(", "&#40;")
                line = line.replace(")", "&#41;")
                line = line.replace("$", "&#36;")
            lines[i] = "<br>" + line

    return "".join(lines)



@app.post("/chat")
async def chat(request: Request):
    global llm, model_config, sampling_params, tokenizer, feature_extractor, tts, llm_embedding
    task_history = []
    query = ''
    start_time = time.time()
    # data = await request.form()
    # query = data.get("query")
    # vis_file = data.get("vis_file")
    # audio_file = data.get("audio_file")
    # text_return = data.get("text_return")

    # print("{}Stage I: get request data {} {}".format('-'*15,time.time() - start_time,'-'*15))
    # if text_return and text_return in ('false','False'):
    #     text_return = False
    # elif text_return and text_return in ('true','True'):
    #     text_return = True
    # elif text_return is None:
    #     text_return = True
    data = await request.form()
    query = data.get("query", "")
    vis_file = data.get("vis_file")
    audio_file = data.get("audio_file")
    text_return = data.get("text_return", "True")  # 默认值设为"True"
    print(f"{'-'*15}Stage I: get request data {time.time() - start_time:.4f} {'-'*15}")

    # 处理 text_return 参数
    text_return = str(text_return).strip().lower() == 'true'
    
    # 判断是否包含视觉和音频文件
    vision_tag = bool(vis_file and vis_file.filename)
    audio_tag = bool(audio_file and audio_file.filename)

    # print("{}{}{}".format('#'*15,query,'#'*15))
    print("{}{}{}".format('#'*15,vis_file,'#'*15))
    # print("{}{}{}".format('#'*15,audio_file,'#'*15))
    tp_1 = time.time()
    print("{}Stage I: processing request data {} {}".format('-'*15,tp_1 - start_time,'-'*15))
    def predict(task_history):
        # chat_query = task_history[-1][0]
        print('{}{}{}'.format('Task History item -1','#'*25,task_history))

        conv_mode = "qwen2p5_instruct"
        conv = conv_templates[conv_mode].copy()
        
        all_audio_path = []
        all_visual_tensor = []

        qs = ''
        input_mode = 'lang'
        for i, (q, a) in enumerate(task_history):
            # if q is a list of tuple, then q should be a file path namely vision/audio
            # if q is audio or text, then a should not be None and thus such q is used as 
            # termination signal for each round in a multi-round conversations. 
            if isinstance(q, (tuple, list)): # include image or video or audio
                if isinstance(q[0],(str, pathlib.PosixPath)): # path string
                    if is_image(q[0]):
                        img_s = time.time()
                        images = [Image.open(q[0]).convert("RGB")]
                        all_visual_tensor.extend(images)
                        input_mode = 'image'
                        qs += DEFAULT_IMAGE_TOKEN * len(images) + '\n'
                        img_end = time.time()
                        print("{}Stage II: processing image time {} {}".format('-'*15,img_end - img_s,'-'*15))
                    elif is_video(q[0]):    
                        vide_s = time.time()         
                        video_frames, slice_len = _get_rawvideo_dec(q[0])
                        all_visual_tensor.extend(video_frames)
                        input_mode = 'video'
                        qs += DEFAULT_IMAGE_TOKEN * slice_len + '\n'
                        vide_e = time.time()
                        print("{}Stage II: processing video time {} {}".format('-'*15,vide_e - vide_s,'-'*15))
                    elif is_wav(q[0]):

                        if a is not None and a.startswith('☜'):
                            continue
                        else:
                            all_audio_path.append(q[0])
                            new_q = qs + DEFAULT_AUDIO_TOKEN
                            qs = ''
                            conv.append_message(conv.roles[0], new_q)
                            conv.append_message(conv.roles[1], a)
                else:   #do not save the image and audio locally and implemented here
                    if isinstance(q[0],Image.Image):
                        images = [q[0]]
                        all_visual_tensor.extend(images)
                        input_mode = 'image'
                        qs += DEFAULT_IMAGE_TOKEN * len(images) + '\n'
                    else:
                        # to implement directly use audio data instead of save locally.
                        raise "Not implemented for video and audio"
            else:  ## pure text qa
                new_q = qs + q
                qs = ''
                conv.append_message(conv.roles[0], new_q)
                conv.append_message(conv.roles[1], a)

        prompt = conv.get_prompt(input_mode)

        if all_audio_path != []:
            input_ids = tokenizer_image_audio_token(
                prompt, tokenizer, 
                image_token_index=model_config.image_token_index, 
                audio_token_index=model_config.audio_token_index
            )
            audio_list = []
            for single_audio_path in all_audio_path:
                try:
                    audio, original_sr = torchaudio.load(single_audio_path)
                    # The FeatureExtractor was trained using a sampling rate of 16000 Hz
                    target_sr = 16000
                    # Resample
                    if original_sr != target_sr:
                        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                        audio = resampler(audio)
                    audio_features = feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt")["input_features"]
                    audio_list.append(audio_features.squeeze(0))
                except Exception as e:
                    print(f"Error processing {single_audio_path}: {e}")
        else:
            input_ids = tokenizer_image_token(
                prompt, tokenizer, 
                image_token_index=model_config.image_token_index
            )



        if all_visual_tensor == [] and all_audio_path == []:
            datapromt={
                 "prompt_token_ids": input_ids,
            }
 
        elif all_visual_tensor != [] and all_audio_path == []:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "image": all_visual_tensor
                    },
            }
        elif all_visual_tensor == [] and all_audio_path != []:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "audio": audio_list
                    },
            }
        else:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "image": all_visual_tensor,
                    "audio": audio_list
                    },
            }
        tp_2 = time.time()
        print("{}Stage II: processing prompt token ids {} {}".format('-'*15,tp_2 - tp_1,'-'*15))
        output = llm.generate(datapromt, sampling_params=sampling_params)
        outputs = output[0].outputs[0].text
        tp_3 = time.time()
        print("{}Stage III: generation process cost {} {}".format('-'*15,tp_3 - tp_2,'-'*15))
        # task_history[-1] = (chat_query, outputs)
        remove_special_characters_output = remove_special_characters(outputs)  
        # _chatbot[-1] = (chat_query, _parse_text(remove_special_characters_output))
        # print("query",chat_query)
        # print("task_history",task_history)
        # print("chatbot: ",_chatbot)
        print("{}Final: total time for request {} {}".format('-'*15,time.time() - start_time,'-'*15))
        print("answer:  ",outputs)
        return remove_special_characters_output

    def add_text(task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        task_history = task_history + [(task_text, None)]
        return task_history

    def add_file( task_history, file):
        task_history = task_history + [((file.name,), None)]
        return task_history

    def add_audio(task_history, file):
        print(file)
        if file is None:
            return  task_history
        task_history = task_history + [((file,), None)]
        return  task_history

    def add_video( task_history, file):
        if file is None:
            return  task_history
        new_file_name = str(file).replace(".webm",".mp4")
        if str(file).endswith(".webm"):
            convert_webm_to_mp4(file, new_file_name)
        task_history = task_history + [((new_file_name,), None)]
        return task_history
    def add_image(task_history, file: UploadFile):
        if file is None:
            return task_history
        image =  Image.open(file.file).convert('RGB')
        task_history = task_history + [((image,), None)]
        return task_history


    def stream_audio_output(outputs):
        print(f"input text {outputs}")
        # text = ''.join(outputs)
        text = outputs
        print(f'join text {text}')
        # import pdb; pdb.set_trace()
        if not text:
            import pdb;pdb.set_trace()
            yield None,None
        llm_resounse = replace_equation(text)
        #print('tts_text', llm_resounse)
        start_time = time.time()
        sample_rate = 24000
        for idx, text in enumerate(split_into_sentences(llm_resounse)):
            embeddings = llm_embedding(torch.tensor(tokenizer.encode(text)).to(device))
            for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), decoder_topk,
                                None, 
                                codec_chunk_size, codec_padding_size):
                if idx == 0:
                    try:
                        split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                        seg = seg[:, :, split_idx:]
                    except:
                        print('Do not need to split')
                        pass
      
                if seg is not None and len(seg) > 0:
                    seg = seg.to(torch.float32).cpu().numpy()
                    end_time = time.time()
                    print(text)
                    print(seg.shape)
                    print('tts_time',end_time - start_time)
                    audio_buffer = io.BytesIO()
                    
                    # wavfile.write(f,sample_rate,float_to_int16(seg).squeeze(0).squeeze(0))
                    soundfile.write(audio_buffer, seg.squeeze(0).squeeze(0), sample_rate, format='wav')
                    audio_buffer.seek(0)
                    # yield f.read() 

                    # torchaudio.save(audio_buffer,torch.tensor(seg.squeeze(0)), sample_rate, format='wav')   
                    # audio_buffer.seek(0)
                    
                    yield audio_buffer.getvalue()   
       
    async def _save_file(file, path = pathlib.Path('userdata')):
        res = await file.read()#读取文件完成后，再进行后面的操作
        ext = file.filename.split('.')[-1]
        filemd5 = hashlib.md5(res).hexdigest()+ '.' + ext#计算md5并返回字符串
        with open(path.joinpath(filemd5), "wb") as f:
            f.write(res)#写入文件，文件名即md5值
            # print(path.joinpath(filemd5))
        filepath=path.joinpath(filemd5)
        return filepath
    save_s = time.time()
    if vision_tag: 
        if is_image(vis_file.filename) or is_video(vis_file.filename):
            filepath = await _save_file(vis_file)
            task_history = add_video(task_history, filepath)
            
            # task_history = add_image(task_history, vis_file)
        else:
            return JSONResponse(content={"Error": "The Vision file is not supported"}, status_code=status.HTTP_400_BAD_REQUEST)

    if (query and not audio_tag) or (not query and audio_tag):
        if query:         ## Text submit       
            task_history = add_text(task_history, query)
        else:             ## Audio submit
            audio_filepath = await _save_file(audio_file)
            task_history = add_audio(task_history, audio_filepath)
    
        # import pdb;pdb.set_trace()
    else:
        return JSONResponse(content={"Error": "Please submit one and only one of text or audio"}, status_code=status.HTTP_400_BAD_REQUEST)
     
    save_e = time.time()
    print("{}Time cost for save files locally{} {}".format('-'*15,save_e - save_s,'-'*15))
    
    if text_return:
        # return StreamingResponse(predict(task_history))
        return JSONResponse(content={'response':predict(task_history)})

    return EventSourceResponse(stream_audio_output(predict(task_history)),media_type='audio/wav')
    # return StreamingResponse(stream_audio_output(predict(task_history)),media_type='audio/wave')

def load_model(model_path = './demo_VITA_ckpt/'):
    global llm, model_config, sampling_params, tokenizer, feature_extractor, tts, llm_embedding

    llm_embedding = load_model_embemding(model_path).to(device)
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={'image':256,'audio':50}
    )  

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=512, best_of=1, skip_special_tokens=False, repetition_penalty=1.2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, subfolder="feature_extractor", trust_remote_code=True)
    tts = llm2TTS(os.path.join(model_path, 'vita_tts_ckpt'))
    # _launch_demo(llm, model_config, sampling_params, tokenizer, feature_extractor, tts, llm_embedding)
    # uvicorn.run(app=app, host="0.0.0.0", port=8000, log_level="info",reload = True)


if __name__ == '__main__':
    
    uvicorn.run(app="web_demo.app_lit:app", host="0.0.0.0", port=7860, log_level="debug",reload=True,reload_dirs='./web_demo',reload_includes='.py')
    # uvicorn.run(app="app_lit:app", host="0.0.0.0", port=7860, log_level="debug",reload=True,reload_dirs='./',reload_includes='.py')


