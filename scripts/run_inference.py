import argparse
import torch

from llava_phi.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_phi.conversation import conv_templates, SeparatorStyle
from llava_phi.model.builder import load_pretrained_model
from llava_phi.utils import disable_torch_init
from llava_phi.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from io import BytesIO

from tqdm import tqdm
import os
import json

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


parser = argparse.ArgumentParser()
parser.add_argument("--image-folder", type=str, default="")
parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
parser.add_argument("--answers-file", type=str, default="answer.jsonl")
args = parser.parse_args()

# Model
disable_torch_init()

model_name = get_model_name_from_path("./model/LLaVA-Med-Phi-rad-v2")
tokenizer, model, image_processor, context_len = load_pretrained_model("./model/LLaVA-Med-Phi-rad-v2", None, model_name)

eval_file_path = args.question_file
img_path = args.image_folder

questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

predict = []

for question in tqdm(questions, total=len(questions)):
    qs=question['text']
    
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    if 'phi' in model_name.lower():
        conv_mode = "phi-2_v0"
    else:
        conv_mode = "default"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_path=img_path+question['image']
    image = load_image(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=tokenizer.eos_token_id,  # Pad token
            use_cache=True,
        )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    predict.append({'answer': outputs, 'question_id': question['question_id']})

with open(args.answers_file,'w') as f:
    json.dump(predict, f)



    
