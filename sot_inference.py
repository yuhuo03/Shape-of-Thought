import os
import json
from datetime import datetime
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO
import re
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae

# Set paths for your trained checkpoint
checkpoint_dir = "your/path/to/models/BAGEL-SoT"
checkpoint_file = "your/path/to/model.safetensors"
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)


print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"GPU memory per device:")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

# LLM config preparing (use base model configs)
llm_config = Qwen2Config.from_json_file(os.path.join(checkpoint_dir, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing (use base model configs)
vit_config = SiglipVisionConfig.from_json_file(os.path.join(checkpoint_dir, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading (use base model VAE)
vae_model, vae_config = load_ae(local_path=os.path.join(checkpoint_dir, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

# Create model with empty weights - IMPORTANT: Use float32 initially to match checkpoint
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing (use base model tokenizer)
tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint_dir)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 512, 14)

# Device mapping for 8x80GB GPUs - use bf16 directly
max_mem_per_gpu = "80GiB"

print("Setting up device mapping...")
device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    dtype=torch.bfloat16,  # Use bf16 for device mapping
)

print("Device map:", device_map)

# Handle same-device modules
same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    if first_device is not None:
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

print("Final device map:", device_map)

# Load checkpoint directly in bf16
print(f"Loading checkpoint directly in bfloat16: {checkpoint_path}")
print("Loading model from safetensors file...")

# Load model directly in bf16
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=checkpoint_path,
    device_map=device_map,
    offload_buffers=False,
    dtype=torch.bfloat16,   # Load directly as bf16
    force_hooks=True,
)

model = model.eval()

print('Model loaded directly in bfloat16!')
print(f"Model dtype: {next(model.parameters()).dtype}")
print("Model loading completed successfully!")

# Check memory usage
print("GPU memory usage after loading:")
for i in range(torch.cuda.device_count()):
    if torch.cuda.memory_allocated(i) > 0:
        allocated = torch.cuda.memory_allocated(i) / 1e9
        cached = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

# Rest of inference code
from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

inference_hyper=dict(
    do_sample=True,
    text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

SOT_SYSTEM_PROMPT = '''You are an AI assistant specialized in shape understanding and 3D object construction. Think step by step through the process of building 3D objects from textual descriptions. Generate visual aids at each construction step to show the progressive assembly of the object. Wrap your reasoning with <think></think> tokens, and wrap your final assembly confirmation with <assembly></assembly> tokens. Provide your final conclusion clearly in the format of '<assembly>Final Assembly: <answer here></assembly>'''

# SoT example prompts (you can modify these or load from SoT dataset)
sot_prompts = [
    "Build a rectangular container with a wide open top, thick vertical sides, and a recessed base, featuring smooth edges and a uniformly solid appearance.",
    "Construct a compact round planter with a sturdy cylindrical base filled with soil, topped by tall, slender leaves extending upward in a clustered arrangement.",
    "Create a rounded ceramic pot with a smooth, bulbous body and a thick, rolled rim circling the wide opening.",
    "Construct a sleek, S-shaped chair with a continuous curved base forming a seamless transition into the seat and back, featuring a single smooth surface design without visible joints or separations.",
    "Construct a modern, minimalist chair featuring a continuous S-shaped form with a single surface seat seamlessly connecting to a vertically aligned backrest, both supported by a flat, rectangular base.",
    "Construct a three-headed chandelier with curved lamp arms extending from a central unit, each supporting a wide, tapered lampshade, connected by a vertical chain for suspension.",
    "Construct a tall, rectangular storage cabinet with five horizontal drawers, each featuring a sleek, curved handle. The cabinet has a flat top panel, vertical side panels, and a solid back panel, forming a sturdy, box-like structure. The base is supported by a series of base side panels, providing stability and a uniform appearance.",
    "Construct a modern chair with a sleek curved single surface seat seamlessly attached to a matching curved backrest, supported by a uniquely arched base that flows naturally from the seat, creating a continuous and minimalist design.",
    "Construct a rectangular ping pong table featuring a smooth, flat tabletop surface with a centered net divider, supported by four tapered legs connected by bar stretchers for stability.",
    "Construct a modern faucet with an angular, L-shaped spout, mounted on a rectangular base, featuring two hexagonal switches on either side of the spout for controlling water flow."
]


# Select a prompt (or load from your dataset)
prompt = sot_prompts[0]  # You can change this index or load from dataset
image = None  # SoT typically starts without an initial image, but you can provide one if needed

print(f"SoT Prompt: {prompt}")
print('-'*50)

# Create output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"reasoning_output_{timestamp}"
images_folder = os.path.join(output_folder, "images")
os.makedirs(images_folder, exist_ok=True)

# Save the original problem images if they exist
problem_image_paths = []
if image is not None:
    if isinstance(image, list):
        # Handle multiple images
        for i, img in enumerate(image):
            problem_image_path = os.path.join(images_folder, f"problem_image_{i+1}.png")
            relative_path = os.path.join("images", f"problem_image_{i+1}.png")
            img.save(problem_image_path)
            problem_image_paths.append(relative_path)
            print(f"Problem image {i+1} saved at '{problem_image_path}'")
    else:
        # Handle single image
        problem_image_path = os.path.join(images_folder, "problem_image.png")
        relative_path = os.path.join("images", "problem_image.png")
        image.save(problem_image_path)
        problem_image_paths.append(relative_path)
        print(f"Problem image saved at '{problem_image_path}'")

reasoning_text = []
reasoning_images = []
image_paths = []  # Store relative paths to images

# Initialize with the SoT prompt
current_input = [prompt]

# Loop until no more vision_start tokens
iteration = 0
while True:
    # Get understanding output
    print(f"iteration: {iteration}")
    output = inferencer.interleave_inference(current_input, understanding_output=True, system_prompt=SOT_SYSTEM_PROMPT, **inference_hyper)

    # Extract the reasoning text
    if '<|im_start|>' in output[0] and '<|im_end|>' in output[0]:
        extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
    else:
        extracted_text = output[0]

    # Clean up the text - extract content from <think> tags if present
    extracted_text = extracted_text.strip()
    if '<think>' in extracted_text and '</think>' in extracted_text:
        # Extract content between <think> tags
        think_start = extracted_text.find('<think>') + len('<think>')
        think_end = extracted_text.find('</think>')
        extracted_text = extracted_text[think_start:think_end].strip()

    # Check for stopping conditions - look for <assembly> tags
    has_final_answer = '<assembly>' in extracted_text.lower() or 'final assembly:' in extracted_text.lower()

    if has_final_answer:
        # Clean up the final answer text by removing image placeholders and assembly tags
        clean_final_answer = re.sub(r'<image_start>\[.*?\]<image_end>', '', extracted_text).strip()
        # Remove <assembly> tags and extract content
        clean_final_answer = re.sub(r'</?assembly>', '', clean_final_answer).strip()
        # Clean up extra whitespace
        clean_final_answer = re.sub(r'\s+', ' ', clean_final_answer).strip()
        reasoning_text.append(clean_final_answer)
        print(f"{clean_final_answer}")
        current_input = current_input + [clean_final_answer]
        break

    reasoning_text.append(extracted_text)
    print(f"{extracted_text}")

    # Generate image based on current reasoning
    current_input_with_reasoning = current_input + [extracted_text]
    output = inferencer.interleave_inference(current_input_with_reasoning, system_prompt=SOT_SYSTEM_PROMPT, **inference_hyper)
    image_output = output[0]

    # Save and collect the generated image
    reasoning_images.append(image_output)
    image_filename = f'reasoning_image_{iteration + 1}.png'
    image_path = os.path.join(images_folder, image_filename)
    relative_image_path = os.path.join("images", image_filename)  # Relative path for JSON

    image_output.save(image_path)
    image_paths.append(relative_image_path)
    print(f"Image saved at '{image_path}'")

    # Update input for next iteration
    current_input = current_input_with_reasoning + [image_output]

    iteration += 1
    print('-'*50)

# Generate final complete image after reasoning is complete
print("Generating final complete image...")
final_image_output = inferencer.interleave_inference(
    [prompt] + reasoning_images,  # Use original prompt with all reasoning images as context
    system_prompt=SOT_SYSTEM_PROMPT,
    **inference_hyper
)[0]

final_image_path = os.path.join(images_folder, "final_complete.png")
final_image_output.save(final_image_path)
print(f"Final image saved at '{final_image_path}'")

# Save reasoning data to JSON
reasoning_data = {
    "timestamp": timestamp,
    "prompt": prompt,
    "system_prompt": SOT_SYSTEM_PROMPT,
    "problem_image_paths": problem_image_paths if problem_image_paths else None,
    "response": [
        {
            "step": i + 1,
            "text": text,
            "image_path": image_paths[i] if i < len(image_paths) else None
        }
        for i, text in enumerate(reasoning_text)
    ],
    "total_steps": len(reasoning_text),
    "total_images": len(image_paths) + 1  # +1 for final image
}

# Save JSON file
json_path = os.path.join(output_folder, "reasoning_data.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(reasoning_data, f, indent=2, ensure_ascii=False)

print(f"\nReasoning complete!")
print(f"Output folder: {output_folder}")
print(f"JSON metadata: {json_path}")
print(f"Generated {len(image_paths)} step images, 1 final image, and {len(reasoning_text)} reasoning steps")

