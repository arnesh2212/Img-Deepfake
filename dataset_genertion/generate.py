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

from google import genai
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from google.genai import types

load_dotenv(Path(".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)



import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.utils import get_precision

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

rank = 32  # you can also use rank=128 model to improve the quality

# -------------------------
# Module-level pipeline init
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
        # load transformer and text encoder (paths from your snippet)
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{PRECISION}_r32-flux.1-krea-dev.safetensors",
            offload=True
        )
        transformer.set_attention_impl("nunchaku-fp16")

        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
            "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors",
            offload=True
        )

        PIPELINE = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-krea-dev",
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        PIPELINE.enable_sequential_cpu_offload()
        print(f"[INFO] Flux pipeline loaded (precision={PRECISION}).")
    except Exception as e:
        _PIPELINE_LOAD_ERROR = e
        PIPELINE = None
        PRECISION = None
        print(f"[ERROR] Could not load Flux pipeline at import: {e}")
    
_safe_load_pipeline()


def _aspect_to_size(aspect_ratio_str, max_dim=1024):
    """
    Convert aspect string like '16:9' to (width, height) with the longer side equal to max_dim.
    Ensure width and height are divisible by 8.
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


def flux_gemini_image(prompt, save_path=None):
    """
    Generate an image using the preloaded FLUX pipeline.
    Signature unchanged: generate_gemini_image(prompt, save_path=None)
    Returns saved file path (str) or None on failure.
    """
    print(f"[INFO] Generating image for prompt:\n{prompt}\n")

    # if pipeline failed to load at import, avoid attempting generation
    if PIPELINE is None:
        print("[ERROR] FLUX pipeline is not available. See earlier import-time error.")
        if _PIPELINE_LOAD_ERROR:
            print(f"[ERROR-DETAIL] {_PIPELINE_LOAD_ERROR}")
        return None

    aspect_ratio_choices = ["16:9", "4:3", "3:2", "1:1", "9:16", "3:4", "2:3", "5:4", "4:5"]
    random_aspect_ratio = random.choice(aspect_ratio_choices)
    width, height = _aspect_to_size(random_aspect_ratio, max_dim=1024)

    # hyperparameters (tweakable but kept internal)
    guidance_scale = 4.5
    num_inference_steps = 28

    try:
        print(f"[INFO] Using Flux pipeline (precision={PRECISION}) with size {width}x{height}")
        out = PIPELINE(prompt, height=height, width=width, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

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



SYSTEM_INSTRUCTION =  """You are an expert real-world photographic prompt engineer. You will receive exactly one short caption (one non-empty line). Convert that caption into a single, natural photographic prompt that reads like an ordinary handheld snapshot a person would actually take — casual, imperfect, and reliably reproducible. Prioritize physical plausibility: obey gravity, scale, perspective, materials, and real-world dynamics. If the caption conflicts with physics or real-world proportions, simplify or change it to a plausible alternative that preserves the user's intent.

ASSUMPTION: If input contains multiple lines, use only the first non-empty line as the caption and ignore the rest.

MANDATORY FORMAT & OUTPUT (follow exactly)
- Produce **exactly one line** and **nothing else**.
- That single line must begin with: `Prompt:` (capital P, colon) followed by one single-sentence photographic prompt.
- Do NOT output lists, multiple prompts, numbered options, examples, code blocks, or additional commentary.
- Do NOT include newline characters; the entire output must be a single line.
- Prefer **25–80 words**; if exceeded, condense while preserving realism.
- End the sentence with the realism limiter: “no heavy bokeh, no cinematic or studio effects, no fake-looking elements, natural handheld snapshot.”

PHYSICS & PROPORTION RULES (follow exactly)

1) Gravity & dynamics
   - Objects and subjects must obey gravity and show physically consistent support and deformation (a backpack sags on shoulders, fabric drapes around hips, a wet towel clings heavy, leaves bend under a frog’s weight).
   - For motion, describe believable motion cues (blur direction, displaced dust, kicked-up sand) consistent with the force and mass involved.

2) Scale & reference objects
   - Always anchor object/animal size to familiar references: “bench-height,” “doorway,” “bicycle wheel,” “adult hand,” “shoe-length,” or “child-sized.” Replace precise numeric specs with natural approximations: “about 1–2 m,” “small,” “hand-sized.”
   - If caption specifies sizes that conflict with nearby objects, correct them to match the reference object.

3) Perspective, depth & occlusion
   - Ensure consistent perspective: foreground objects larger, background smaller; include occlusion or contact (finger touching object, paw pressing grass) to avoid floating subjects.
   - Mention shadows/reflections that match light direction and viewpoint to anchor subjects in space.

4) Materials & surface behavior
   - Describe realistic material responses: wet surfaces darken and reflect, sand holds footprints, metal shows specular highlights and small scratches, fur lies flat or fluff depending on wind.
   - Avoid unrealistic textures (plastic skin, waxy fur); prefer natural texture descriptors (matte skin, coarse fur, damp feathers).

5) Animal & human interactions (safety + scale)
   - Enforce species-appropriate distances, posture, and contact. Large wild animals must not be within arm’s reach of unprotected people. If close interaction is required, add plausible context (handler, leash, enclosure).
   - Ensure animal size aligns with proximate objects (bench, hand, car).

6) Lighting coherence
   - State realistic lighting and ensure it matches shadows/reflections and material properties (“late-afternoon warm light casting long shadows to the left,” “overcast soft light with diffuse shadows”).

7) Realism verification (internal checklist)
   - Before finalizing, mentally confirm: (a) gravity and support are plausible, (b) scale matches reference objects, (c) shadows/reflections match light, (d) materials react plausibly to environment, (e) moving elements show consistent motion cues, (f) no floating/merged/duplicated anatomy. If any check fails, adjust the scene (change distance, add contact, swap for realistic object, or add handler/leash/fence).

CAPTION FLEXIBILITY (always apply)
- Treat caption as guidance, not a literal script. If it contains impossible or overly specific details, keep the main intent but simplify: pick 1–2 strongest realistic details, replace exact numbers with approximations, or swap improbable elements for plausible alternatives.

TONE & STRUCTURE
- Single natural photographic sentence. Prefer this order:
  subject → action/pose/dynamics → clothing/objects/material reactions → immediate foreground/contact → midground → background → lighting/time/weather → subtle artifacts/imperfections → realism limiter.

OUTPUT LITERAL ENDING (must be present)
- Finish the single sentence with: “no heavy bokeh, no cinematic or studio effects, no fake-looking elements, natural handheld snapshot.”

REMINDER (must be followed)
- Output exactly one single line only, beginning with `Prompt:` and followed by one single-sentence prompt that obeys these physics, scale, and realism rules. No other text.
"""

load_dotenv(Path(".env"))

GEMINI_API_KEY_TIER1=os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_TIER0=os.getenv("GEMINI_API_KEY_TIER0")

GROQ_API_KEY=os.getenv("GROQ_API_KEY")



class prompt_gemini:
    def __init__(self,caps_list:str,model_name:str="gemini-2.5-flash-lite"):
        """
        Args:
            caps_list(str): Each caption is separated by a line space. Each caption starts with \" and ends with \"
        """
        self.sys_inst= SYSTEM_INSTRUCTION
        
        self.client=genai.Client(
        api_key=GEMINI_API_KEY_TIER0
        )

        self.model = model_name
        self.text="""your list of captions is as follows (one caption occupies only one line):"""+caps_list
        
        self.contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=self.text),
                    ],
                ),
            ]
        self.generate_content_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(
                thinking_budget=-1,
                ),
            system_instruction=[
                types.Part.from_text(text=self.sys_inst),
                ],
            )

    def generate(self)->str:
        prompts=""
        for chunk in self.client.models.generate_content_stream(
        model=self.model,
        contents=self.contents,
        config=self.generate_content_config,
        ):
            #print(chunk.text,end="")
            prompts+=chunk.text
        
        return prompts
    


from groq import Groq

class prompt_groq:
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


GROQ_MODELS = ["llama-3.1-8b-instant","qwen/qwen3-32b","moonshotai/kimi-k2-instruct"]
probs = {"gemini":0.6, "groq":0.4}
probs_grq = {"llama-3.1-8b-instant":0.4,"qwen/qwen3-32b":0.3, "moonshotai/kimi-k2-instruct" :0.3}

import ast

new_df = "done_dataset_v1.csv"
cols = ["synset", "caption", "prompt", "image_filename", "prompt_model_used"]
folder = "flux_generated_images"

if not os.path.exists(folder):
   os.makedirs(folder)

if __name__ == "__main__":
    df = pd.read_csv("flickr_annotations_30k.csv")
    #Reverse df
   # Extract first caption from JSON-like list in 'raw' column
    def _first_caption(raw_val):
        try:
            if pd.isna(raw_val):
                return ""
            parsed = ast.literal_eval(raw_val)
            return parsed[0] if parsed else ""
        except Exception:
            # fallback: return raw string (or empty)
            return str(raw_val)

    df["captions"] = df["raw"].apply(_first_caption)

    # Ensure synset is string to match done_df's synset dtype later
    df["synset"] = df["img_id"].astype(str)

    # Reverse df
    df = df.iloc[::-1].reset_index(drop=True)
    print(f"Total captions to process: {len(df)}")
    print(df.head(5))
    print(df.columns)

    if os.path.isfile(new_df):
        done_df = pd.read_csv(new_df)

        # Ensure done_df['synset'] is string too so merge keys match types
        if "synset" in done_df.columns:
            done_df["synset"] = done_df["synset"].astype(str)

        merged = df.merge(
            done_df[['synset', 'caption']],
            left_on=['synset', 'captions'],
            right_on=['synset', 'caption'],
            how='left',
            indicator=True
        )

        df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge', 'caption'])
        print(f"Captions remaining after skipping done ones: {len(df)}")
        print(f"Total captions to process: {len(df)}")

    # --- MAIN LOOP ---
    for index, row in df.iterrows():
        synset = row["synset"]
        caption = row["captions"]
        prompt_model_used = ""
        image_filename = (
            "FLICKR-"
            + synset
            + "-flux-"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"-{index}.jpg"
        )

        # Skip if already in new_df
        if os.path.isfile(new_df):
            done_df = pd.read_csv(new_df)
            if ((done_df["synset"] == synset) & (done_df["caption"] == caption)).any():
                print(f"Skipping caption index {index} as it is already processed.")
                continue

        model_choice = random.choices(list(probs.keys()), weights=list(probs.values()))[0]

        if model_choice == "gemini":
            try:
                prompt_gen = prompt_gemini(caps_list="\"" + caption + "\"")
                prompt = prompt_gen.generate()
                prompt_model_used = "gemini-2.5-flash-lite"
            except Exception as e:
                print(f"Gemini API error: {e}. Falling back to Groq.")
                groq_model_choice = "llama-3.1-8b-instant"
                prompt_gen = prompt_groq(caps_list="\"" + caption + "\"", model_name=groq_model_choice)
                prompt = prompt_gen.generate()
                prompt_model_used = groq_model_choice
        else:
            try:
                try:
                    groq_model_choice = random.choices(
                        list(probs_grq.keys()), weights=list(probs_grq.values())
                    )[0]
                    prompt_gen = prompt_groq(
                        caps_list="\"" + caption + "\"", model_name=groq_model_choice
                    )
                    prompt = prompt_gen.generate()
                    prompt_model_used = groq_model_choice
                except Exception as e:
                    model = "llama-3.1-8b-instant"
                    prompt_gen = prompt_groq(caps_list="\"" + caption + "\"", model_name=model)
                    prompt = prompt_gen.generate()
                    prompt_model_used = model
            except Exception as e:
                print(f"Groq API error: {e}. Falling back to Gemini.")
                prompt_gen = prompt_gemini(caps_list="\"" + caption + "\"", model_name="gemma-3-27b-it")
                prompt = prompt_gen.generate()
                prompt_model_used = "gemini-2.5-flash-lite"

        # Clean prompt
        prompt = re.sub(r"<think>.*?</think>", "", prompt, flags=re.DOTALL).strip()

        image_path = os.path.join(folder, image_filename)
        try:
            gen_image_msg = flux_gemini_image(prompt, save_path=image_path)
            print(f"Generated image for caption index {index}: {gen_image_msg}")

            if os.path.isfile(image_path):
                print(f"Image successfully generated and saved at {image_path}")
            else:
                print(
                    f"Image generation failed for caption index {index}. No file found at {image_path}"
                )
                continue

            # Append new record to CSV
            new_row = pd.DataFrame(
                [[synset, caption, prompt, image_filename, prompt_model_used]], columns=cols
            )
            if not os.path.isfile(new_df):
                new_row.to_csv(new_df, index=False)
            else:
                new_row.to_csv(new_df, mode="a", header=False, index=False)

        except Exception as e:
            print(f"Error generating image for caption index {index}: {e}")