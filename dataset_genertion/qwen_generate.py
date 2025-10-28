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
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import requests
from datetime import datetime
from datasets import load_dataset # Changed import
import math

from google import genai
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from google.genai import types

load_dotenv(Path(".env"))
from huggingface_hub import hf_hub_download
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# -------------------------
# Module-level pipeline init
# (UNCHANGED)
# -------------------------
PIPELINE = None
PRECISION = None
_PIPELINE_LOAD_ERROR = None

def _safe_load_pipeline():
    global PIPELINE, PRECISION, _PIPELINE_LOAD_ERROR
    if PIPELINE is not None or _PIPELINE_LOAD_ERROR is not None:
        return

    try:
        PRECISION = get_precision()  # auto-detect 'int4' or 'fp4'
        
        # Scheduler config for Lightning
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": True,
            "stochastic_sampling": True,        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        
        num_inference_steps = 8  # Lightning 4-step
        rank = 32
        model_paths = {
            4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{PRECISION}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
            8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{PRECISION}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
        }
        
        # Load transformer
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[num_inference_steps])
        
        PIPELINE = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image",
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )
        
        # # Memory optimization
        # if get_gpu_memory() > 18:
        #     PIPELINE.enable_model_cpu_offload()
        # else:
        #     transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        #     PIPELINE._exclude_from_cpu_offload.append("transformer")
        #     PIPELINE.enable_sequential_cpu_offload()
        PIPELINE.to("cuda")
            
        print(f"[INFO] Qwen-Image Lightning pipeline loaded (precision={PRECISION}).")
    except Exception as e:
        _PIPELINE_LOAD_ERROR = e
        PIPELINE = None
        PRECISION = None
        print(f"[ERROR] Could not load Qwen pipeline at import: {e}")
    
_safe_load_pipeline()


def _aspect_to_size(aspect_ratio_str, max_dim=1024):
    """
    (UNCHANGED)
    """
    try:
        a, b = aspect_ratio_str.split(":")
        a, b = int(a), int(b)
        if a >= b:
            width = max_dim
            height = int(round(max_dim * (b / a)))
        else:
            height = max_dim
            width = int(round(max_dim * (a / b)))
        # make divisible by 8
        width = max(8, (width // 8) * 8)
        height = max(8, (height // 8) * 8)
        return width, height
    except Exception:
        return max_dim, max_dim


def qwen_image(prompt, save_path=None):
    """
    (UNCHANGED)
    """
    print(f"[INFO] Generating image for prompt:\n{prompt}\n")

    # if pipeline failed to load at import, avoid attempting generation
    if PIPELINE is None:
        print("[ERROR] Qwen pipeline is not available. See earlier import-time error.")
        if _PIPELINE_LOAD_ERROR:
            print(f"[ERROR-DETAIL] {_PIPELINE_LOAD_ERROR}")
        return None

    aspect_ratio_choices = ["16:9", "4:3", "3:2", "1:1", "9:16", "3:4", "2:3", "5:4", "4:5"]
    random_aspect_ratio = random.choice(aspect_ratio_choices)
    width, height = _aspect_to_size(random_aspect_ratio, max_dim=1024)

    try:
        print(f"[INFO] Using Qwen Lightning pipeline (precision={PRECISION}) with size {width}x{height}")
        out = PIPELINE(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=8,
            true_cfg_scale=1.0
        )

        # Extract PIL Image
        img = None
        if hasattr(out, "images") and out.images:
            img = out.images[0]
        elif isinstance(out, list) and out:
            img = out[0]
        else:
            print("[WARN] Pipeline returned no images.")
            return None

        filename = save_path

        img.save(filename)
        print(f"[SAVED] Image saved at {filename}")
        return filename

    except Exception as e:
        print(f"[ERROR] Failed to generate/save image: {e}")
        return None


from io import BytesIO
from PIL import Image
import requests

def download_real_image(url, save_path, compress_quality=85, max_edge=1024):
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



SYSTEM_INSTRUCTION = """(UNCHANGED)"""

load_dotenv(Path(".env"))

GEMINI_API_KEY_TIER1 = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_TIER0 = os.getenv("GEMINI_API_KEY_TIER0")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class prompt_gemini:
    # (UNCHANGED)
    def __init__(self, caps_list: str, model_name: str = "gemini-2.5-flash-lite"):
        """
        Args:
            caps_list(str): Each caption is separated by a line space. Each caption starts with \" and ends with \"
        """
        self.sys_inst = SYSTEM_INSTRUCTION
        
        self.client = genai.Client(
            api_key=GEMINI_API_KEY_TIER0
        )

        self.model = model_name
        self.text = """your list of captions is as follows (one caption occupies only one line):""" + caps_list
        
        self.contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=self.text),
                ],
            ),
        ]
        self.generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
            system_instruction=[
                types.Part.from_text(text=self.sys_inst),
            ],
        )

    def generate(self) -> str:
        prompts = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=self.contents,
            config=self.generate_content_config,
        ):
            prompts += chunk.text
        
        return prompts


from groq import Groq


class prompt_groq:
    # (UNCHANGED)
    def __init__(self, caps_list: str, model_name: str = "openai/gpt-oss-20b"):
        """
        Args:
            caps_list (str): Each caption is separated by a line space.
                             Each caption starts with \" and ends with \"
        """
        self.sys_inst = SYSTEM_INSTRUCTION
        self.text = "Your list of captions is as follows (one caption occupies only one line):\n" + caps_list
        self.model = model_name
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self) -> str:
        prompts = ""

        # Groq client uses OpenAI-compatible chat completions API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_inst},
                {"role": "user", "content": self.text},
            ],
            temperature=0.8,
            max_tokens=2048,
            stream=True,
        )

        for chunk in response:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                print(delta.content, end="")
                prompts += delta.content

        return prompts


GROQ_MODELS = ["llama-3.1-8b-instant", "qwen/qwen3-32b", "moonshotai/kimi-k2-instruct"]
probs = {"gemini": 0.6, "groq": 0.4}
probs_grq = {"llama-3.1-8b-instant": 0.4, "qwen/qwen3-32b": 0.3, "moonshotai/kimi-k2-instruct": 0.3}

new_df = "done_dataset_qwen.csv"
cols = ["image_id", "caption", "prompt", "real_filename", "deepfake_filename", "prompt_model_used"]
folder_real = "real_pexels"
folder_deepfake = "deepfake_pexels"

if not os.path.exists(folder_real):
    os.makedirs(folder_real)
if not os.path.exists(folder_deepfake):
    os.makedirs(folder_deepfake)

if __name__ == "__main__":
    # Load dataset - first 50k only
    print("[INFO] Loading dataset...")
    
    # --- START MODIFICATION ---
    # Load the new dataset from Hugging Face
    ds = load_dataset("CaptionEmporium/pexels-568k-internvl2", split='train')
    df = ds.to_pandas()
    # --- END MODIFICATION ---
    
    # Take only first 50k
    df = df.head(50000)
    
    # --- START MODIFICATION ---
    # Define function to choose caption with 70/30 probability
    def choose_caption(row):
        if random.random() < 0.7:  # 70% chance for cogvlm_caption
            # Use cogvlm_caption if it's valid, otherwise fall back to internvl2_caption
            if row['cogvlm_caption'] and str(row['cogvlm_caption']).strip():
                return row['cogvlm_caption']
            else:
                return row['internvl2_caption']
        else:  # 30% chance for internvl2_caption
            # Use internvl2_caption if it's valid, otherwise fall back to cogvlm_caption
            if row['internvl2_caption'] and str(row['internvl2_caption']).strip():
                return row['internvl2_caption']
            else:
                return row['cogvlm_caption']

    # Apply the function to create the new 'captions' column
    df["captions"] = df.apply(choose_caption, axis=1)
    
    # Use '__index_level_0__' as the unique image ID
    df["image_id"] = df["__index_level_0__"].astype(str)
    
    # Clean up: drop rows where the chosen caption is None or empty
    df = df.dropna(subset=['captions'])
    df = df[df['captions'].str.strip() != '']
    # --- END MODIFICATION ---

    # Reverse df
    df = df.iloc[::-1].reset_index(drop=True)
    print(f"Total captions to process: {len(df)}")
    print(df.head(5))
    print(df.columns)

    if os.path.isfile(new_df):
        done_df = pd.read_csv(new_df)

        # Ensure done_df['image_id'] is string too so merge keys match types
        if "image_id" in done_df.columns:
            done_df["image_id"] = done_df["image_id"].astype(str)

        # --- START MODIFICATION ---
        # Update merge logic to use the new 'captions' column from df
        merged = df.merge(
            done_df[['image_id', 'caption']],
            left_on=['image_id', 'captions'], # 'captions' is the new column
            right_on=['image_id', 'caption'],  # 'caption' is the column name in the CSV
            how='left',
            indicator=True
        )
        # --- END MODIFICATION ---

        df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge', 'caption'])
        print(f"Captions remaining after skipping done ones: {len(df)}")
        print(f"Total captions to process: {len(df)}")

    # --- MAIN LOOP ---
    for index, row in df.iterrows():
        # --- START MODIFICATION ---
        # Map new dataset columns to existing variables
        image_id = row["image_id"]  # From '__index_level_0__'
        caption = row["captions"]   # From the randomly selected caption
        url = row["url"]            # From 'url'
        # --- END MODIFICATION ---
        
        prompt_model_used = ""
        
        real_filename = f"{image_id}-real.jpg"
        deepfake_filename = f"{image_id}-qwen-{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        real_path = os.path.join(folder_real, real_filename)
        deepfake_path = os.path.join(folder_deepfake, deepfake_filename)

        # Enhanced resume logic: skip if both files exist
        if os.path.exists(real_path) and os.path.exists(deepfake_path):
            print(f"[SKIP] Both real and deepfake exist for {image_id}")
            continue

        # Skip if already in new_df with both files
        if os.path.isfile(new_df):
            done_df = pd.read_csv(new_df)
            # Ensure image_id is string for comparison
            done_df["image_id"] = done_df["image_id"].astype(str)
            
            if ((done_df["image_id"] == image_id) & (done_df["caption"] == caption)).any():
                existing_row_matches = done_df[(done_df["image_id"] == image_id) & (done_df["caption"] == caption)]
                
                if not existing_row_matches.empty:
                    existing_row = existing_row_matches.iloc[0]
                    existing_real = os.path.join(folder_real, existing_row["real_filename"])
                    existing_deepfake = os.path.join(folder_deepfake, existing_row["deepfake_filename"])
                    
                    if os.path.exists(existing_real) and os.path.exists(existing_deepfake):
                        print(f"[SKIP] Caption index {index} already processed and files exist.")
                        continue
                    else:
                        print(f"[RETRY] Caption index {index} found in CSV but files missing. Retrying...")
                

        # Download real image if not exists
        if not os.path.exists(real_path):
            print(f"[INFO] Downloading real image for {image_id}...")
            success = download_real_image(url, real_path)
            if not success:
                print(f"[ERROR] Failed to download real image for {image_id}. Skipping...")
                continue

        # (Prompt generation logic UNCHANGED)
        # model_choice = random.choices(list(probs.keys()), weights=list(probs.values()))[0]

        # if model_choice == "gemini":
        #     try:
        #         prompt_gen = prompt_gemini(caps_list="\"" + caption + "\"")
        #         prompt = prompt_gen.generate()
        #         prompt_model_used = "gemini-2.5-flash-lite"
        #     except Exception as e:
        #         print(f"Gemini API error: {e}. Falling back to Groq.")
        #         groq_model_choice = "llama-3.1-8b-instant"
        #         prompt_gen = prompt_groq(caps_list="\"" + caption + "\"", model_name=groq_model_choice)
        #         prompt = prompt_gen.generate()
        #         prompt_model_used = groq_model_choice
        # else:
        #     try:
        #         try:
        #             groq_model_choice = random.choices(
        #                 list(probs_grq.keys()), weights=list(probs_grq.values())
        #             )[0]
        #             prompt_gen = prompt_groq(
        #                 caps_list="\"" + caption + "\"", model_name=groq_model_choice
        #             )
        #             prompt = prompt_gen.generate()
        #             prompt_model_used = groq_model_choice
        #         except Exception as e:
        #             model = "llama-3.1-8b-instant"
        #             prompt_gen = prompt_groq(caps_list="\"" + caption + "\"", model_name=model)
        #             prompt = prompt_gen.generate()
        #             prompt_model_used = model
        #     except Exception as e:
        #         print(f"Groq API error: {e}. Falling back to Gemini.")
        #         prompt_gen = prompt_gemini(caps_list="\"" + caption + "\"", model_name="gemma-3-27b-it")
        #         prompt = prompt_gen.generate()
        #         prompt_model_used = "gemini-2.5-flash-lite"

        # (Prompt cleaning logic UNCHANGED)
        # prompt = re.sub(r"<think>.*?</think>", "", prompt, flags=re.DOTALL).strip()
        
        # (Simple prompt logic UNCHANGED)
        prompt = caption + "Safe for work, ultra-realistic, lifelike skin textures, minor imperfections, natural pores, realistic hair strands, natural lighting, subtle film grain, candid handheld snapshot, no plastic or waxy skin, not over-smoothed, detailed"
        try:
            # (Image generation call UNCHANGED)
            gen_image_msg = qwen_image(prompt, save_path=deepfake_path)
            print(f"Generated image for caption index {index}: {gen_image_msg}")

            if os.path.isfile(deepfake_path):
                print(f"Image successfully generated and saved at {deepfake_path}")
            else:
                print(
                    f"Image generation failed for caption index {index}. No file found at {deepfake_path}"
                )
                continue

            # (CSV saving logic UNCHANGED)
            new_row = pd.DataFrame(
                [[image_id, caption, prompt, real_filename, deepfake_filename, prompt_model_used]], columns=cols
            )
            if not os.path.isfile(new_df):
                new_row.to_csv(new_df, index=False)
            else:
                new_row.to_csv(new_df, mode="a", header=False, index=False)

        except Exception as e:
            print(f"Error generating image for caption index {index}: {e}")