"""
FIXED: Better Prompt Parsing and Format Enforcement

Handles:
- <think> tags from reasoning models
- Multiple separators (;;, ;;; etc)
- Retry logic if format is wrong
- Cleaner prompt extraction
"""

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
    """Build system instruction with STRONG format enforcement"""
    
    category_text = "\n\n".join([
        CATEGORY_INSTRUCTIONS[cat] for cat in selected_categories
    ])
    
    system_prompt = f"""You are an expert image manipulation prompt engineer. Generate THREE distinct manipulation prompts.

**AVAILABLE CATEGORIES** (use ONE per prompt, all different):

{category_text}

**CRITICAL OUTPUT FORMAT - FOLLOW EXACTLY:**

You MUST output EXACTLY in this format:
Edit the image to: [first manipulation] ;; Edit the image to: [second manipulation] ;; Edit the image to: [third manipulation]

**RULES:**
1. Start each prompt with "Edit the image to: "
2. Separate prompts with " ;; " (space-semicolon-semicolon-space)
3. NO <think> tags, NO explanations, NO numbering
4. Each prompt: 15-50 words
5. Make edits BOLD and NOTICEABLE
6. Use DIFFERENT category for each prompt

**EXAMPLE:**
Edit the image to: Replace the woman's face with a man's face of similar age ;; Edit the image to: Add bright red sunglasses with visible reflections ;; Edit the image to: Change the indoor background to an outdoor beach with palm trees

Image caption: {{caption}}

Output ONLY the three prompts in the format above, nothing else."""
    
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
        self.client = genai.Client(api_key=GEMINI_API_KEY_TIER0 or GEMINI_API_KEY)
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
                        temperature=0.8,  # Slightly lower for better format adherence
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
        self.client = Groq(api_key=GROQ_API_KEY)
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
                    temperature=0.8,
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

def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    """Download an image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img.save(save_path, 'JPEG', quality=95)
        print(f"[DOWNLOAD SUCCESS]")
        return True
        
    except Exception as e:
        print(f"[DOWNLOAD ERROR] {e}")
        return False


def edit_gemini_image(original_image_path: str, edit_prompt: str, save_path: str):
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


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_jsonl_dataset(
    jsonl_path: str,
    output_dir: str = "deepfake_dataset",
    csv_output: str = "deepfake_metadata.csv",
    max_images: int = None,
    download_delay: float = 1.0,
    api_delay: float = 2.0
):
    """COMPLETE pipeline with better parsing"""
    
    os.makedirs(output_dir, exist_ok=True)
    original_dir = os.path.join(output_dir, "original")
    edited_dir = os.path.join(output_dir, "edited")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)
    
    columns = [
        'image_id', 'original_caption', 'original_url', 'original_image_path',
        'selected_categories', 'category_1', 'category_2', 'category_3',
        'edit_prompt_1', 'edited_image_path_1', 'edit1_success',
        'edit_prompt_2', 'edited_image_path_2', 'edit2_success',
        'edit_prompt_3', 'edited_image_path_3', 'edit3_success',
        'prompt_model_used', 'download_success', 'timestamp'
    ]
    
    # Load progress
    if os.path.exists(csv_output):
        done_df = pd.read_csv(csv_output)
        processed_urls = set(done_df['original_url'].tolist())
        print(f"[INFO] Found {len(processed_urls)} already processed images")
    else:
        processed_urls = set()
        pd.DataFrame(columns=columns).to_csv(csv_output, index=False)
    
    # Load JSONL
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"[INFO] Total images: {len(data)}\n")
    
    if max_images:
        data = data[:max_images]
    
    # Model probabilities - PREFER GEMINI (better format adherence)
    model_probs = {"gemini": 0.8, "groq": 0.2}
    probs_grq = {
        "llama-3.1-8b-instant": 0.7,  # Most reliable
        "qwen/qwen3-32b": 0.15,  # Sometimes has <think> issues
        "moonshotai/kimi-k2-instruct": 0.15
    }
    
    category_usage = {i: 0 for i in range(1, 8)}
    
    for idx, item in enumerate(data):
        url = item.get('url', '')
        caption = item.get('caption_llava', '') or item.get('caption_llava_short', '')
        
        if not caption or not url or url in processed_urls:
            continue
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(data)}]")
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = f"CC12M-human-nano-banana-{timestamp_str}-{idx}"
        
        selected_categories = sample_categories(n=3, weights=CATEGORY_WEIGHTS)
        print(f"[CATEGORIES] {selected_categories}")
        
        for cat in selected_categories:
            category_usage[cat] += 1
        
        result = {
            'image_id': image_id,
            'original_caption': caption,
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
            'timestamp': timestamp_str
        }
        
        try:
            # DOWNLOAD
            original_filename = f"{image_id}_original.jpg"
            original_path = os.path.join(original_dir, original_filename)
            download_success = download_image(url, original_path)
            result['download_success'] = download_success
            result['original_image_path'] = original_path
            
            if not download_success:
                pd.DataFrame([result]).to_csv(csv_output, mode='a', header=False, index=False)
                continue
            
            time.sleep(download_delay)
            
            # GENERATE PROMPTS
            print(f"[PROMPTS] Generating...")
            
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
                    print(f"[WARN] Gemini failed, using Groq")
                    groq_model = "llama-3.1-8b-instant"  # Most reliable
                    prompt_gen = TamperingPromptGroq(caption, selected_categories, groq_model)
                    prompts_text = prompt_gen.generate()
                    prompt_model = groq_model
            else:
                groq_model = random.choices(
                    list(probs_grq.keys()),
                    weights=list(probs_grq.values())
                )[0]
                prompt_gen = TamperingPromptGroq(caption, selected_categories, groq_model)
                prompts_text = prompt_gen.generate()
                prompt_model = groq_model
            
            result['prompt_model_used'] = prompt_model
            
            # PARSE PROMPTS
            prompts = clean_and_parse_prompts(prompts_text)
            
            # Ensure we have exactly 3
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
                edited_filename = f"{image_id}_edit{i}.jpg"
                edited_path = os.path.join(edited_dir, edited_filename)
                
                try:
                    edit_result = edit_gemini_image(original_path, prompt, edited_path)
                    
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
    
    print(f"\n{'='*80}")
    print(f"[CATEGORY USAGE]")
    total = sum(category_usage.values())
    for cat in sorted(category_usage.keys()):
        count = category_usage[cat]
        pct = (count / total * 100) if total > 0 else 0
        print(f"  Cat {cat}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, default="deepfake_dataset")
    parser.add_argument("--csv", type=str, default="deepfake_metadata.csv")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--download-delay", type=float, default=1.0)
    parser.add_argument("--api-delay", type=float, default=2.0)
    
    args = parser.parse_args()
    
    process_jsonl_dataset(
        jsonl_path=args.jsonl,
        output_dir=args.output,
        csv_output=args.csv,
        max_images=args.max,
        download_delay=args.download_delay,
        api_delay=args.api_delay
    )