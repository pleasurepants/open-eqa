import os
import json
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import glob

model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

DATASET_DIR = "/nfs/data2/zhang/openeqa/hm3d/data/frames"
INPUT_JSON = "/home/wiss/zhang/code/open-eqa/data/open-eqa-hm3d.json"
OUTPUT_JSON = "/home/wiss/zhang/code/open-eqa/Llava-1.5/llava-hm3d.json"

K = 50

def sample_frames(frame_list, k):
    total_frames = len(frame_list)
    if total_frames == 0:
        return []
    if total_frames <= k:
        return frame_list
    indices = [int(i * total_frames / k) for i in range(k)]
    return [frame_list[i] for i in indices]

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

for idx, item in enumerate(tqdm(data, desc="Generating captions")):
    video_folder = item["episode_history"]
    folder_path = os.path.join(DATASET_DIR, video_folder)

    frame_list = sorted(glob.glob(os.path.join(folder_path, "*-rgb.png")))
    if not frame_list:
        print(f"Warning: No frames found in {folder_path}. Skipping...")
        continue

    selected_frames = sample_frames(frame_list, K)
    item["answers"] = []

    # prompt = f"""You are an intelligent question answering agent. I will ask you questions about an indoor space and you must provide a short answer.
    #             You will be shown a set of images collected from a single location.
    #             Given a user query, you must output `text` to answer the question asked by the user.
    #             Answer concisely in one or two words.
    #             User Query: {item['question']} <image>
    #             Answer:"""
    prompt= f"""You are an intelligent caption generate agent. Describe concisely in one sentence what is shown in the image. Mention key objects, their attributes, and spatial relationships if applicable. <image> Answer:"""

    for frame_path in selected_frames:
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                do_sample=False
            )

        caption = processor.decode(output[0], skip_special_tokens=True).split("Answer: ")[1].strip()
        item["answers"].append({"frame": os.path.basename(frame_path), "caption": caption})

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated captions saved incrementally to {OUTPUT_JSON}")
