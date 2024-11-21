import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import re
import logging
import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import uvicorn
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()


def build_json(response, time):
    return {"assistant": response,
            "time": f"{time:.2f}"}


def log_response(uuid, request, answer_json, prompt, model_output):
    logger.info(f"---------------------- uuid: {uuid} ----------------------\n"
                f"json request {request}\n"
                f"prompt {prompt}\n"
                f"output {model_output}\n"
                f"answer {answer_json}")


@app.post("/interact")
async def interact(request: Request):
    json_obj = None
    try:
        start_time = datetime.datetime.now().timestamp()
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
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = llm.generate([text], sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        messages += [{"role": "assistant", "content": generated_text}]
        history[user_id] = messages
        response_time = datetime.datetime.now().timestamp() - start_time
        answer = build_json(generated_text, response_time)
        log_response(user_id, json_obj, answer, text, generated_text)
        return answer
    except Exception:
        logger.exception(f'JSON data: {json_obj}')
        raise HTTPException(status_code=100, detail="Internal Error")


logger = logging.getLogger("default_logger")
logger.setLevel(logging.INFO)
today = datetime.datetime.today().date().strftime('%Y-%m-%d')
os.makedirs('logs/qwen-14b', exist_ok=True)
fh = logging.FileHandler(f'logs/qwen-14b/{today}.txt', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
logger.addHandler(fh)

sampling_params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=4000, logprobs=0, n=1, use_beam_search=False)

model_name_or_path = '/home/wulingkeji/.cache/modelscope/hub/qwen/Qwen1___5-14B-Chat-AWQ/'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=4000, quantization="AWQ")
history = {}

uvicorn.run(app, host='0.0.0.0', port=8592, workers=0)
