"""
FIXED: Better Prompt Parsing and Format Enforcement

Handles:
- <think> tags from reasoning models
- Multiple separators (;;, ;;; etc)
- Retry logic if format is wrong
- Cleaner prompt extraction
"""
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from datasets import load_dataset
from sys import prefix
import pandas as pd
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import random
from PIL import Image
from io import BytesIO
import re
from datetime import datetime
import json
import requests
import time
import numpy as np
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# Load API keys
load_dotenv(Path(".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_TIER0 = os.getenv("GEMINI_API_KEY_TIER0")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================================
# CATEGORY WEIGHTING
# ============================================================================

CATEGORY_WEIGHTS = {
    1: 0.30,  # Face Manipulations
    2: 0.25,  # Object Addition/Removal
    3: 0.20,  # Clothing & Appearance
    5: 0.03,  # Background Manipulation
    4: 0.05,  # Lighting & Shadows
    6: 0.15,  # Body & Pose
    7: 0.02,  # Context Inconsistencies 
}
# sum of abovbe probab = 0.3 +0.25 +0.2 +0.03 +0.05 +0.15 +0.02 = 1.0
def sample_categories(n=3, weights=CATEGORY_WEIGHTS):
    """Sample n unique categories based on weighted distribution"""
    categories = list(weights.keys())
    probs = np.array(list(weights.values()))
    probs = probs / np.sum(probs)
    
    selected = np.random.choice(
        categories, 
        size=n, 
        replace=False, 
        p=probs
    )
    
    return selected.tolist()


# ============================================================================
# CATEGORY INSTRUCTIONS
# ============================================================================

CATEGORY_INSTRUCTIONS = {
    1: """**Category 1: FACE MANIPULATIONS**
- Swap the person's face with a different person's face
- Change facial expression dramatically (neutral to laughing, serious to shocked)
- Change facial features (different nose, eye color, jaw structure)
- Add or remove facial hair significantly
- Age the person significantly (15-30 years older/younger)""",
    
    2: """**Category 2: OBJECT ADDITION/REMOVAL**
- Add visible accessories: glasses, sunglasses, jewelry, hat, scarf
- Remove existing accessories 
- Add objects in hands: smartphone, drink, bag, food
- Remove objects from hands
- Add people in background
- Remove people or objects from scene""",
    
    3: """**Category 3: CLOTHING & APPEARANCE**
- Change clothing color completely
- Change outfit style entirely (casual to formal, summer to winter)
- Add or remove visible tattoos
- Change hairstyle drastically or hair color
- Change entire clothing item""",
    
    4: """**Category 4: LIGHTING & SHADOWS**
- Add inconsistent shadows (wrong direction)
- Change lighting dramatically on person vs background
- Add/remove reflections
- Make lighting mismatches obvious""",
    
    5: """**Category 5: BACKGROUND MANIPULATION**
- Replace entire background (indoor→outdoor, city→beach, etc)
- Add multiple people in background
- Remove major background elements
- Change time of day in background
- Swap location completely""",
    
    6: """**Category 6: BODY & POSE MODIFICATIONS**
- Change body proportions noticeably
- Modify posture significantly
- Add or remove limbs in group photos
- Change skin tone visibly""",
    
    7: """**Category 7: CONTEXT INCONSISTENCIES**
- Season mismatch (winter clothes in summer setting)
- Scale problems (oversized objects)
- Add modern items to old photos
- Weather mismatches"""
}


def build_system_instruction(selected_categories):

    category_text = "\n\n".join([
        CATEGORY_INSTRUCTIONS[cat] for cat in selected_categories
    ])

    system_prompt = f"""You are an expert image manipulation prompt engineer. Generate THREE distinct edit prompts that are **strictly PHOTOREALISTIC**, **physically plausible**, and **faithful to the original image**.
    ```

    **OBJECTIVE — REALISTIC EDITS ONLY:**
    All edits must look like real photographs taken in the real world. **Do NOT** produce any stylized, genre, artistic, or fantasy edits. Prohibited: "cyberpunk", "anime", "cinematic", "surreal", "glowing", "futuristic", "magical", "painting", "CGI", or anything visually implausible.

    **AVAILABLE CATEGORIES** (use ONE per prompt, all different):

    {category_text}

    **STRICT OUTPUT FORMAT (follow exactly):**
    Edit the image to: [first manipulation] ;; Edit the image to: [second manipulation] ;; Edit the image to: [third manipulation]

    **REALISM RULES (MANDATORY):**

    1. Edits must be physically and visually consistent — correct perspective, lighting, and shadows.
    2. Do not change unrelated parts of the image. Modify only what the edit logically requires.
    3. Background changes are allowed only to realistic settings (e.g., park, street, kitchen, office). No stylized or imaginary environments.
    4. Preserve subject identity and proportions; avoid celebrity likenesses or impossible anatomy.
    5. Describe materials, textures, and light naturally (e.g., “soft daylight”, “warm evening glow”, “reflections on glass”).
    6. Edits should be **plausible but noticeable** — realistic photography adjustments, not artistic transformations.
    7. Each prompt must be 15–50 words.
    8. Use a **different category** for each prompt.
    9. **Output only the three prompts**, exactly in the specified format — no numbering, no explanations.

    **EXAMPLE (good):**
    Edit the image to: Replace the man’s jacket with a navy cotton blazer showing realistic fabric texture and soft window light ;; Edit the image to: Remove background clutter and place the subject on a quiet urban street with natural shadows ;; Edit the image to: Add subtle morning sunlight from the right, giving warm highlights on hair and shoulders

    Image caption: {{caption}}

    Output **only** the three prompts in the format above.
    """
    return system_prompt



# ============================================================================
# IMPROVED PROMPT PARSING
# ============================================================================

def clean_and_parse_prompts(text):
    """
    Aggressively clean and parse prompts from LLM output
    
    Handles:
    - <think> tags and reasoning
    - Multiple separator variants
    - Extra whitespace
    - Numbered lists
    """
    
    # Step 1: Remove <think> tags and everything inside
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Step 2: Remove common reasoning patterns
    text = re.sub(r'(?:Okay|Alright|Let me|First|Second|Third|Now|Here)[,:].*?(?=Edit the image|$)', '', text, flags=re.IGNORECASE)
    
    # Step 3: Extract only lines starting with "Edit the image"
    edit_lines = re.findall(r'Edit the image to:.*?(?=(?:Edit the image to:|;;|$))', text, flags=re.DOTALL | re.IGNORECASE)
    
    if edit_lines:
        # Clean each line
        prompts = []
        for line in edit_lines:
            # Remove separator remnants
            line = re.sub(r';;+', '', line)
            # Clean whitespace
            line = ' '.join(line.split())
            # Ensure it starts correctly
            if not line.lower().startswith('edit the image to:'):
                line = 'Edit the image to: ' + line
            prompts.append(line.strip())
        
        return prompts[:3]  # Take first 3
    
    # Step 4: Fallback - split by ;; and clean
    parts = re.split(r';;+', text)
    prompts = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Extract the actual edit instruction
        # Look for "Edit the image to:" or similar
        match = re.search(r'(?:Edit the image to:)(.*?)(?:;;|$)', part, flags=re.IGNORECASE | re.DOTALL)
        if match:
            prompt = "Edit the image to:" + match.group(1).strip()
            prompts.append(prompt)
        elif 'edit' in part.lower() and len(part) > 20:
            # Try to salvage if it looks like a prompt
            if not part.lower().startswith('edit the image'):
                part = 'Edit the image to: ' + part
            prompts.append(part)
    
    return prompts[:3]


# ============================================================================
# PROMPT GENERATION WITH RETRY
# ============================================================================

class TamperingPromptGemini:
    def __init__(self, caption: str, selected_categories: list, model_name: str = "gemini-2.5-flash-lite"):
        self.sys_inst = build_system_instruction(selected_categories)
        self.client = genai.Client(api_key=gemini_api_keys[gem_api_ind])
        self.model = model_name
        self.caption = caption
        self.selected_categories = selected_categories
        
    def generate(self, max_retries=2):
        """Generate with retry logic"""
        user_message = self.sys_inst.replace('{caption}', self.caption)
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[user_message],
                    config=types.GenerateContentConfig(
                        temperature=0.5,  # Slightly lower for better format adherence
                        top_p=0.95,
                        max_output_tokens=600,
                    )
                )
                
                result = response.text.strip()
                
                # Try to parse
                prompts = clean_and_parse_prompts(result)
                
                if len(prompts) >= 3:
                    return ' ;; '.join(prompts[:3])
                
                # If not enough prompts and not last retry, try again
                if attempt < max_retries - 1:
                    print(f"[RETRY] Only got {len(prompts)} prompts, retrying...")
                    continue
                
                # Last attempt - return what we have
                return ' ;; '.join(prompts + [''] * (3 - len(prompts)))
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"[RETRY] Error: {e}, retrying...")
        
        return ""


class TamperingPromptGroq:
    def __init__(self, caption: str, selected_categories: list, model_name: str = "llama-3.1-8b-instant"):
        from groq import Groq
        
        self.sys_inst = build_system_instruction(selected_categories)
        self.client = Groq(api_key=groq_api_keys[groq_api_ind])
        self.model = model_name
        self.caption = caption
        self.selected_categories = selected_categories
        
    def generate(self, max_retries=2):
        """Generate with retry logic"""
        
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.sys_inst},
                        {"role": "user", "content": f"Image caption: {self.caption}"}
                    ],
                    model=self.model,
                    temperature=0.5,
                    max_tokens=600,
                    top_p=0.95
                )
                
                result = chat_completion.choices[0].message.content.strip()
                
                # Try to parse
                prompts = clean_and_parse_prompts(result)
                
                if len(prompts) >= 3:
                    return ' ;; '.join(prompts[:3])
                
                if attempt < max_retries - 1:
                    print(f"[RETRY] Only got {len(prompts)} prompts, retrying...")
                    continue
                
                return ' ;; '.join(prompts + [''] * (3 - len(prompts)))
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"[RETRY] Error: {e}, retrying...")
        
        return ""


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

#def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
#    """Download an image from URL"""
#    try:
#        headers = {
#            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#        }
#        
#        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
#        response.raise_for_status()
#        
#        img = Image.open(BytesIO(response.content))
#        
#        if img.mode == 'RGBA':
#            img = img.convert('RGB')
#        
#        img.save(save_path, 'JPEG', quality=95)
#        print(f"[DOWNLOAD SUCCESS]")
#        return True
#        
#    except Exception as e:
#        print(f"[DOWNLOAD ERROR] {e}")
#        return False
#
def download_image(url, save_path, compress_quality=85, max_edge=1024):
    """
    Download original image from dataset URL, resize if needed (max edge=1024),
    and save with JPEG compression.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')

        # Resize if the image is larger than max_edge
        width, height = img.size
        max_dim = max(width, height)
        if max_dim > max_edge:
            scale = max_edge / max_dim
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # Save with compression
        img.save(save_path, 'JPEG', quality=compress_quality, optimize=True)

        print(f"[DOWNLOADED & COMPRESSED] Saved at {save_path} (size={img.size}, quality={compress_quality})")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download/compress {url}: {e}")
        return False

def edit_gemini_image(original_image_path: str, edit_prompt: str, save_path: str):  #TODO Change to qwen image edit
    """Edit an image using Gemini"""
    print(f"[EDITING] {edit_prompt[:60]}...")
    
    try:
        original_image = Image.open(original_image_path)
        
        img_byte_arr = BytesIO()
        original_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        full_prompt = f"{edit_prompt}. Maintain photorealistic quality and ensure the edit is clearly visible."
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[
                types.Part(inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=img_byte_arr.read()
                )),
                types.Part(text=full_prompt)
            ],
        )
        
        saved_path = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(save_path)
                print(f"[EDIT SUCCESS]")
                saved_path = save_path
        
        return saved_path
    
    except Exception as e:
        print(f"[EDIT ERROR] {e}")
        return None



import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision


def initialize_qwen_img_edit(rank:int =32):
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    num_inference_steps = 8  # you can also use the 8-step model to improve the quality
    rank = 32  # you can also use the rank=128 model to improve the quality
    model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509-lightningv2.0-{num_inference_steps}steps.safetensors"

    # Load the model
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
    )

    return pipeline.to("cuda")

def edit_qwen_image(pipeline, original_image_path: str, edit_prompt: str, save_path: str):

    try:
        image = load_image(original_image_path)
        image = image.convert("RGB")

        inputs = {
            "image": [image],
            "prompt": edit_prompt,
            "true_cfg_scale": 4.0,
            "negative_prompt": "low quality, unrealistic, distorted, altered background, extra objects, unnecessary changes, artifacts, lowres, blur, bad lighting, unnatural colors",
            "num_inference_steps": 8,
        }

        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(save_path)
        
        saved_path=save_path
        return saved_path
    
    except Exception as e:
        print(f"[EDIT ERROR] {e}")
        return None

# ============================================================================
# MAIN PROCESSING
# ============================================================================




def process_pexels_dataset(
    QWEN_PIPELINE,
    output_dir: str = "deepfake_dataset",
    csv_output: str = "deepfake_metadata.csv",
    max_images: int = 50000,
    download_delay: float = 1.0,
    api_delay: float = 2.0,
    caption_prob_cogvlm: float = 0.1  # 70% cogvlm, 30% internvl2
    
):
    """COMPLETE pipeline for Pexels dataset with weighted categories"""
    
    os.makedirs(output_dir, exist_ok=True)
    original_dir = os.path.join(output_dir, "original")
    edited_dir = os.path.join(output_dir, "edited")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)
    
    columns = [
        'image_id', 'original_caption', 'caption_source', 'original_url', 'original_image_path',
        'selected_categories', 'category_1', 'category_2', 'category_3',
        'edit_prompt_1', 'edited_image_path_1', 'edit1_success',
        'edit_prompt_2', 'edited_image_path_2', 'edit2_success',
        'edit_prompt_3', 'edited_image_path_3', 'edit3_success',
        'prompt_model_used', 'download_success', 'timestamp'
    ]
    
    # Load progress
    if os.path.exists(csv_output):
        done_df = pd.read_csv(csv_output)
        # Convert image_id to string for comparison
        done_df['image_id'] = done_df['image_id'].astype(str)
        processed_ids = set(done_df['image_id'].tolist())
        print(f"[INFO] Found {len(processed_ids)} already processed images")
    else:
        processed_ids = set()
        pd.DataFrame(columns=columns).to_csv(csv_output, index=False)
    
    # Load Pexels dataset from Hugging Face
    print(f"[INFO] Loading Pexels dataset from Hugging Face...")
    ds = load_dataset("CaptionEmporium/pexels-568k-internvl2", split='train')
    df = ds.to_pandas()
    #reverse df for processing order
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Take only first max_images if specified
    if max_images:
        df = df.head(max_images)
    
    print(f"[INFO] Total images in dataset: {len(df)}")
    
    # Caption selection function (70% cogvlm, 30% internvl2)
    def choose_caption(row):
        if random.random() < caption_prob_cogvlm:  # 70% chance for cogvlm_caption
            if row.get('cogvlm_caption') and str(row['cogvlm_caption']).strip():
                return row['cogvlm_caption'], 'cogvlm'
            else:
                return row.get('internvl2_caption', ''), 'internvl2'
        else:  # 30% chance for internvl2_caption
            if row.get('internvl2_caption') and str(row['internvl2_caption']).strip():
                return row['internvl2_caption'], 'internvl2'
            else:
                return row.get('cogvlm_caption', ''), 'cogvlm'
    
    # Apply caption selection
    df[['caption', 'caption_source']] = df.apply(
        lambda row: pd.Series(choose_caption(row)), 
        axis=1
    )
    
    # Use _index_level_0_ as image_id
    # df['image_id'] = df['_index_level_0_'].astype(str)
    
    # Clean up: drop rows with empty captions
    df = df.dropna(subset=['caption'])
    df = df[df['caption'].str.strip() != '']
    
    df = df.iloc[::-1].reset_index(drop=True)
    
    print(f"[INFO] Total images after filtering: {len(df)}")
    print(f"[INFO] Using weighted category distribution:")
    for cat_id, weight in CATEGORY_WEIGHTS.items():
        print(f"  Category {cat_id}: {weight*100:.1f}%")
    print()
    
    # Model probabilities
    model_probs = {"gemini": 0.6, "groq": 0.4}  # Prefer Gemini
    probs_grq = {
        "llama-3.1-8b-instant": 0.4,
        "qwen/qwen3-32b": 0.4,
        "moonshotai/kimi-k2-instruct": 0.3
    }
    
    # Track category usage
    category_usage = {i: 0 for i in range(1, 8)}
    
    for idx, row in df.iterrows():
        image_id = str(row['__index_level_0__']) + "-" + row["class_label"].replace(" ","_")  + "-" + str(row['id'])
        caption = row['caption']
        caption_source = row['caption_source']
        url = row['url']
        
        if not caption or not url or image_id in processed_ids:
            continue
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(df)}] ID: {image_id}")
        print(f"Caption ({caption_source}): {caption[:80]}...")
        
        # Generate filename
        image_filename_prefix = f"PEXELS-{image_id}-qwen-edit"

        # Sample categories
        selected_categories = sample_categories(n=3, weights=CATEGORY_WEIGHTS)
        print(f"[CATEGORIES] {selected_categories}")
        
        for cat in selected_categories:
            category_usage[cat] += 1
        
        result = {
            'image_id': image_id,
            'original_caption': caption,
            'caption_source': caption_source,
            'original_url': url,
            'original_image_path': '',
            'selected_categories': str(selected_categories),
            'category_1': selected_categories[0],
            'category_2': selected_categories[1],
            'category_3': selected_categories[2],
            'edit_prompt_1': '',
            'edited_image_path_1': '',
            'edit1_success': False,
            'edit_prompt_2': '',
            'edited_image_path_2': '',
            'edit2_success': False,
            'edit_prompt_3': '',
            'edited_image_path_3': '',
            'edit3_success': False,
            'prompt_model_used': '',
            'download_success': False,
        }
        
        try:
            # DOWNLOAD
            original_filename = f"{image_filename_prefix}_original.jpg"
            original_path = os.path.join(original_dir, original_filename)
            
            download_success = download_image(url, original_path)
            result['download_success'] = download_success
            result['original_image_path'] = original_path
            
            if not download_success:
                pd.DataFrame([result]).to_csv(csv_output, mode='a', header=False, index=False)
                continue
            
            # time.sleep(download_delay)
            
            # GENERATE PROMPTS
            print(f"[PROMPTS] Generating...")
            
            
            counter=0
            prompts_text,prompt_model=None,None
            while (prompts_text is None and prompt_model is None) and counter<10:
                
                model_choice = random.choices(
                list(model_probs.keys()),
                weights=list(model_probs.values())
                )[0]

                if model_choice == "gemini":
                    try:
                        prompt_gen = TamperingPromptGemini(caption, selected_categories)
                        prompts_text = prompt_gen.generate()
                        prompt_model = "gemini-2.5-flash-lite"
                    except Exception as e:
                        gem_api_ind=(gem_api_ind+1)% len(gemini_api_keys)   #Increment api index to change api key
                        counter+=1
                        #print(f"[WARN] Gemini failed, using Groq")
                        #groq_model = "llama-3.1-8b-instant"
                        #prompt_gen = TamperingPromptGroq(caption, selected_categories, groq_model)
                        #prompts_text = prompt_gen.generate()
                        #prompt_model = groq_model
                else:
                    try:
                        groq_model = random.choices(
                            list(probs_grq.keys()),
                            weights=list(probs_grq.values())
                        )[0]
                        prompt_gen = TamperingPromptGroq(caption, selected_categories, groq_model)
                        prompts_text = prompt_gen.generate()
                        prompt_model = groq_model
                    except Exception as e:
                        groq_api_ind=(groq_api_ind+1)% len(groq_api_keys)   #Increment api index to change api key
                        counter+=1

            result['prompt_model_used'] = prompt_model
            #except Exception:
            
            # PARSE PROMPTS
            prompts = clean_and_parse_prompts(prompts_text)
            
            # Ensure exactly 3
            while len(prompts) < 3:
                prompts.append("")
            prompts = prompts[:3]
            
            print(f"[PROMPTS] Got {len([p for p in prompts if p])} valid prompts")
            for i, p in enumerate(prompts, 1):
                if p:
                    print(f"  [{i}] {p[:70]}...")
                    result[f'edit_prompt_{i}'] = p
            
            time.sleep(api_delay)
            
            # EDIT IMAGES
            for i, prompt in enumerate(prompts, 1):
                if not prompt:
                    continue
                
                print(f"\n[EDIT {i}] Cat{selected_categories[i-1]}")
                edited_filename = f"{image_filename_prefix}_edit{i}.jpg"
                edited_path = os.path.join(edited_dir, edited_filename)
                
                try:
                    #edit_result = edit_gemini_image(original_path, prompt, edited_path)
                    edit_result = edit_qwen_image(QWEN_PIPELINE,original_path, prompt, edited_path)
                    if edit_result and os.path.exists(edited_path):
                        result[f'edited_image_path_{i}'] = edited_path
                        result[f'edit{i}_success'] = True
                    
                    time.sleep(api_delay)
                    
                except Exception as e:
                    print(f"[ERROR] Edit {i}: {e}")
            
            # SAVE
            pd.DataFrame([result]).to_csv(csv_output, mode='a', header=False, index=False)
            print(f"\n[COMPLETE]")
            
        except Exception as e:
            print(f"[ERROR] {e}")
            pd.DataFrame([result]).to_csv(csv_output, mode='a', header=False, index=False)
            continue
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"[FINISHED] Check {csv_output}")
    print(f"\n[CATEGORY USAGE]")
    total = sum(category_usage.values())
    for cat in sorted(category_usage.keys()):
        count = category_usage[cat]
        pct = (count / total * 100) if total > 0 else 0
        print(f"  Cat {cat}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, default="/home/arush/deepfake/Our_dataset/qwen_edit_pexels")
    parser.add_argument("--csv", type=str, default="/home/arush/deepfake/Our_dataset/qwen_edit_pexels_metadata.csv")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--download-delay", type=float, default=1.0)
    parser.add_argument("--api-delay", type=float, default=2.0)
    
    args = parser.parse_args()
    PIPELINE=initialize_qwen_img_edit() #Initialize Pipeline
    gemini_api_keys=["AIzaSyARd-60WICxfPWU15ffrPrTQb-JS754kZ4","AIzaSyB6WtdMlMf98uoNO4_NXfEC3Bik9oIztd0","AIzaSyAAgsHG4R1rp1XR2_d9YHKwDqma6wW5iGs","AIzaSyDTD-vzrLOADyyMd2sguZe5vK8BV5UlXuY","AIzaSyBxXOuEfUISpMFKthnKjVjk3Fm0a3Ydcuo","AIzaSyCdn-UPUT-ZHLGL8rlq7gyUlBTAugumvYg"] #! add more
    groq_api_keys=["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","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","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","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","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","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"] #! add more
    gem_api_ind,groq_api_ind=0,0
    
    process_pexels_dataset(
        PIPELINE,
        output_dir=args.output,
        csv_output=args.csv,
        download_delay=args.download_delay,
        api_delay=args.api_delay
    )