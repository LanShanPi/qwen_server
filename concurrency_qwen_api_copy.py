# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# import re
# import logging
# import datetime
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# import uvicorn
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.concurrency import run_in_threadpool
# import asyncio


# app = FastAPI()

# def build_json(response, time):
#     return {"assistant": response,
#             "time": f"{time:.2f}"}

# def log_response(uuid, request, answer_json, prompt, model_output):
#     logger.info(f"---------------------- uuid: {uuid} ----------------------\n"
#                 f"json request {request}\n"
#                 f"prompt {prompt}\n"
#                 f"output {model_output}\n"
#                 f"answer {answer_json}")

# async def generate_response(text, sampling_params):
#     try:
#         outputs = llm.generate([text], sampling_params, use_tqdm=False)
#         return outputs[0].outputs[0].text
#     except Exception as e:
#         logger.exception("Error in generate_response: %s", e)
#         raise

# # llm_lock = asyncio.Lock()
# # async def generate_response(text, sampling_params):
# #     # async with llm_lock:
# #     try:
# #         outputs = await run_in_threadpool(llm.generate, [text], sampling_params, use_tqdm=False)
# #         return outputs[0].outputs[0].text
# #     except Exception as e:
# #         logger.exception("Error in generate_response: %s", e)
# #         raise


# # async def generate_response(text, sampling_params):
# #     # 默认线程池大小为cpu核数的5倍
# #     # 也可自定义
# #     # from concurrent.futures import ThreadPoolExecutor
# #     # executor = ThreadPoolExecutor(max_workers=50)
# #     # await run_in_threadpool(。。。, executor=custom_executor)
# #     outputs = await run_in_threadpool(llm.generate, [text], sampling_params, use_tqdm=False)
# #     return outputs[0].outputs[0].text

# @app.post("/interact")
# async def interact(request: Request):
#     json_obj = None
#     try:
#         start_time = datetime.datetime.now().timestamp()
#         json_obj = await request.json()
#         logger.debug(json_obj)
#         user_id = json_obj.get("uuid")
#         system_prompt = json_obj.get("system")
#         user_prompt = json_obj.get("user")
#         if user_prompt is None:
#             logger.exception(f'JSON data: {json_obj}')
#             raise HTTPException(status_code=100, detail="User prompt cannot be empty!")
#         if system_prompt is None:
#             messages = history.get(user_id, [])
#         else:
#             messages = [{"role": "system", "content": system_prompt}]
#         messages += [{"role": "user", "content": user_prompt}]
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         generated_text = await generate_response(text, sampling_params)
#         messages += [{"role": "assistant", "content": generated_text}]
#         history[user_id] = messages
#         response_time = datetime.datetime.now().timestamp() - start_time
#         answer = build_json(generated_text, response_time)
#         log_response(user_id, json_obj, answer, text, generated_text)
#         return answer
#     except Exception:
#         logger.exception(f'JSON data: {json_obj}')
#         raise HTTPException(status_code=100, detail="Internal Error")

import os
import re
import logging
import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
import asyncio
import time

# 设置环境变量以指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

app = FastAPI()

# 全局变量
request_queue = []
queue_lock = asyncio.Lock()
batch_size = 8
batch_timeout = 0.0001

def build_json(response, time):
    return {"assistant": response, "time": f"{time:.2f}"}

def log_response(uuid, request, answer_json, prompt, model_output):
    logger.info(f"---------------------- uuid: {uuid} ----------------------\n"
                f"json request {request}\n"
                f"prompt {prompt}\n"
                f"output {model_output}\n"
                f"answer {answer_json}")

async def batch_process():
    while True:
        await asyncio.sleep(batch_timeout)
        async with queue_lock:
            if len(request_queue) == 0:
                continue
            batch_requests = request_queue[:batch_size]
            request_queue[:batch_size] = []
        if not batch_requests:
            continue
        texts = [req['text'] for req in batch_requests]
        try:
            outputs = llm.generate(texts, sampling_params, use_tqdm=False)
            for req, output in zip(batch_requests, outputs):
                req['future'].set_result(output.outputs[0].text)
        except Exception as e:
            for req in batch_requests:
                req['future'].set_exception(e)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_process())

@app.post("/interact")
async def interact(request: Request):
    json_obj = None
    try:
        start_time = time.time()
        json_obj = await request.json()
        logger.debug(json_obj)
        user_id = json_obj.get("uuid")
        system_prompt = json_obj.get("system")
        user_prompt = json_obj.get("user")
        if user_prompt is None:
            logger.exception(f'JSON data: {json_obj}')
            raise HTTPException(status_code=100, detail="User prompt cannot be empty!")
        if system_prompt is None:
            messages = history.get(user_id, [])
        else:
            messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.debug(f"Generated text template: {text}")

        future = asyncio.get_event_loop().create_future()
        async with queue_lock:
            request_queue.append({'text': text, 'future': future})
        
        generated_text = await future

        messages += [{"role": "assistant", "content": generated_text}]
        history[user_id] = messages
        response_time = time.time() - start_time
        answer = build_json(generated_text, response_time)
        log_response(user_id, json_obj, answer, text, generated_text)
        return answer
    except Exception as e:
        logger.exception(f'JSON data: {json_obj}')
        raise HTTPException(status_code=100, detail="Internal Error")

logger = logging.getLogger("default_logger")
logger.setLevel(logging.INFO)
today = datetime.datetime.today().date().strftime('%Y-%m-%d')
os.makedirs('logs/qwen-14b-concurrency', exist_ok=True)
fh = logging.FileHandler(f'logs/qwen-14b-concurrency/{today}.txt', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
logger.addHandler(fh)

sampling_params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=4000, logprobs=0, n=1, use_beam_search=False)

model_name_or_path = '/home/kuaipan/model/qwen/Qwen1___5-14B-Chat-AWQ/'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=4000, quantization="AWQ")
history = {}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8594)

# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen1.5-14B-Chat-AWQ --local-dir /home/kuaipan/model
