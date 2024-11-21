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

# ####################################################


import os
import re
import logging
import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import time
import json


# 设置环境变量以指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
app = FastAPI(debug=True)
# 全局变量
request_queue = []
queue_lock = asyncio.Lock()
batch_size = 24
batch_timeout = 0.1

######公用函数######
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
######公用函数######

###########for leap##############
def build_json(response, time):
    return {"assistant": response, "time": f"{time:.2f}"}
def log_response_(uuid, request, answer_json, prompt, model_output):
    leap_logger.info(f"---------------------- uuid: {uuid} ----------------------\n"
                f"json request {request}\n"
                f"prompt {prompt}\n"
                f"output {model_output}\n"
                f"answer {answer_json}")
@app.post("/interact")
async def interact_leap(request: Request):
    json_obj = None
    try:
        start_time = time.time()
        json_obj = await request.json()
        leap_logger.debug(json_obj)
        user_id = json_obj.get("uuid")
        system_prompt = json_obj.get("system")
        user_prompt = json_obj.get("user")
        
        if user_prompt is None:
            leap_logger.exception(f'JSON data: {json_obj}')
            raise HTTPException(status_code=100, detail="User prompt cannot be empty!")
        if system_prompt is None:
            messages = history.get(user_id, [])
        else:
            messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": "user", "content": user_prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        leap_logger.debug(f"Generated text template: {text}")

        future = asyncio.get_event_loop().create_future()
        async with queue_lock:
            request_queue.append({'text': text, 'future': future})
        
        generated_text = await future

        messages += [{"role": "assistant", "content": generated_text}]
        history[user_id] = messages

        response_time = time.time() - start_time
        answer = build_json(generated_text, response_time)
        log_response_(user_id, json_obj, answer, text, generated_text)
        return answer
    except Exception as e:
        leap_logger.exception(f'JSON data: {json_obj}')
        raise HTTPException(status_code=100, detail="Internal Error")


##########for dify############
def log_response(uuid, request, answer_json, prompt, model_output):
    dify_logger.info(f"---------------------- uuid: {uuid} ----------------------\n"
                f"json request {request}\n"
                f"prompt {prompt}\n"
                f"output {model_output}\n"
                f"answer {answer_json}")
# 定义流式生成的异步生成器
async def generate_streaming_response(future):
    while not future.done():
        await asyncio.sleep(0.1)
    chunks = future.result()
    # chunks = re.split(r'(\n\n|\n)', result)  # 按照换行符拆分输出为多个小块
    for index, chunk in enumerate(chunks):
        data = json.dumps({"id":f"{index}","object":"chat.completion.chunk","created":int(time.time()),"model":"qwen1.5-14b", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":index,"delta":{"content":chunk.strip()},"logprobs":None,"finish_reason":None}]})
        yield f"data: {data}\n\n"
    data = json.dumps({"id":f"{len(chunks)}","object":"chat.completion.chunk","created":int(time.time()),"model":"qwen1.5-14b", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":len(chunks),"delta":{},"logprobs":None,"finish_reason":"stop"}]})
    yield f"data: {data}\n\n"

@app.post("/v1/chat/completions")
async def interact_dify(request: Request):
    json_obj = None
    try:
        start_time = time.time()
        json_obj = await request.json()
        dify_logger.debug(json_obj)
        user_id = json_obj.get("uuid")
        scene = json_obj.get("scene")
        messages = json_obj.get("messages", [])
        
        if len(messages) == 1 and "ping" in messages[0]["content"]:
            response = {
                "id": f"ping-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "qwen25-14b",  # 替换为你的模型名称
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Service is running..."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,  # 简单示例，实际根据输入计算
                    "completion_tokens": 5,  # "Service is running..." 的 token 数
                    "total_tokens": 6
                }
            }
            return response

        if len(messages) <= 1:
            dify_logger.exception(f'JSON data: {json_obj}')
            raise HTTPException(status_code=100, detail="用户输入不能为空!")
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        dify_logger.debug(f"Generated text template: {text}")

        future = asyncio.get_event_loop().create_future()
        async with queue_lock:
            request_queue.append({'text': text, 'future': future})
        
        # 返回流式响应
        response_time = time.time() - start_time
        log_response(user_id, json_obj, "streaming", text, "")
        return StreamingResponse(generate_streaming_response(future), media_type="text/event-stream")

    except Exception as e:
        dify_logger.exception(f'JSON data: {json_obj}')
        raise HTTPException(status_code=100, detail="Internal Error")

# 设置日志记录(dify)
dify_logger = logging.getLogger("dify_logger25")
dify_logger.setLevel(logging.INFO)
today = datetime.datetime.today().date().strftime('%Y-%m-%d')
os.makedirs('logs/qwen25-14b-concurrency_dify', exist_ok=True)
dify_fh = logging.FileHandler(f'logs/qwen25-14b-concurrency_dify/{today}.txt', encoding='utf-8')
dify_fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
dify_logger.addHandler(dify_fh)

# 设置日志记录（LEAP）
history = {}
leap_logger = logging.getLogger("leap_logger25")
leap_logger.setLevel(logging.INFO)
today = datetime.datetime.today().date().strftime('%Y-%m-%d')
os.makedirs('logs/qwen25-14b-concurrency_leap', exist_ok=True)
leap_fh = logging.FileHandler(f'logs/qwen25-14b-concurrency_leap/{today}.txt', encoding='utf-8')
leap_fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
leap_logger.addHandler(leap_fh)

# 设置LLM和采样参数
sampling_params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=4000, logprobs=0, n=1, use_beam_search=False)
model_name_or_path = '/home/kuaipan/model/qwen/Qwen2_5-14B-Instruct-AWQ'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=4000, quantization="AWQ")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8592)
