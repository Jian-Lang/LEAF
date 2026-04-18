import argparse
import os
from pathlib import Path

import pandas as pd
import unsloth
from icecream import ic
from loguru import logger
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from unsloth import FastVisionModel


class LMM_model:
    def __init__(self, load_path):
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=load_path,  # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit=True,  # Set to False for 16bit LoRA
            max_seq_length=8192 * 4,
        )
        FastVisionModel.for_inference(self.model)

    def chat(self, inputs):
        output_ids = self.model.generate(**inputs, max_new_tokens=2048, use_cache=True)
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text


class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_df = pd.read_json(f"data/{dataset_name}/data.jsonl", lines=True)
        self.test_vids = pd.read_csv(f"data/{dataset_name}/vids/test.csv").iloc[:, 0].tolist()
        self.data_df = self.data_df[self.data_df["vid"].isin(self.test_vids)]
        self.data_df = self.data_df.sample(frac=1, random_state=2025).reset_index(drop=True)

    def __len__(self):
        return len(self.data_df)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.data_df):
            raise StopIteration
        row = self.data_df.iloc[self.index]
        vid = row["vid"]
        title = row["title"]
        transcript = row["transcript"][:10000]
        ocr = row["ocr"][:10000]
        label = row["label"]
        image_paths = [f"data/{self.dataset_name}/frames_16/{vid}/frame_{i:03d}.jpg" for i in range(0, 16)]
        images = image_paths
        if True:
            # if "4img" in kargs.get("mark", ""):
            images = [image_paths[0], image_paths[4], image_paths[8], image_paths[12]]

        self.index += 1
        return {
            "vid": vid,
            "title": title,
            "transcript": transcript,
            "ocr": ocr,
            "label": label,
            "images": images,
        }


def detection_evaluate(dataset_name, result_path, load_path, num_img, **kargs):
    ablation = kargs.get("ablation", "")
    llm_model = LMM_model(load_path)
    dataset_loader = DatasetLoader(dataset_name)
    save_path = result_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and save_path.stat().st_size > 0:
        logger.info(f"Loading results from {save_path}")
        save_df = pd.read_json(save_path, lines=True)
        # save_df = pd.DataFrame(columns=["vid", "label", "explanation", "prediction"])
    else:
        save_df = pd.DataFrame(columns=["vid", "label", "explanation", "prediction"])
    save_vids = save_df["vid"].values
    bar = tqdm(dataset_loader, total=len(dataset_loader))
    acc, f1 = 0.0, 0.0
    for sample in bar:
        images = sample["images"]

        vid = sample["vid"]
        # bar.set_description(f"vid: {vid}, acc: {acc:.4f}, f1: {f1:.4f}")
        if vid in save_vids:
            continue
        title = sample["title"]
        transcript = sample["transcript"]
        ocr = sample["ocr"]
        label = sample["label"]
        images = [Image.open(path).convert("RGB") for path in images]

        input_text = f"""
You are a moderator on a video platform.

You are provided with the following inputs from a video:
- 16 consecutive video frames (visual context),
- the video title,
- the full transcript,
- any on-screen text extracted via OCR.

Video Title: [{title}]
Video Transcript: [{transcript}]
On-screen Text: [{ocr}]

Hateful video definition: The video is classified as hateful as it contains hateful content targeting individuals or groups based on race, religion, gender, sexuality, or other identities.

Your task is to jointly consider all the inputs and incorporate contextual background knowledge of the video to understand the nuanced sentiment of the content.  
Based on this, judge whether the video is hateful or benign.
Provide a clear and easy-to-understand reasoning for your judgment.

Respond in the following format:
Thought: [your reasoning in natural language]  
Answer: [hateful/benign]
"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * num_img + [{"type": "text", "text": input_text}],
            }
        ]
        if ablation == "label-only":
            input_text = input_text.replace(
                "Provide a clear and easy-to-understand reasoning for your judgment.", ""
            )
            input_text = input_text.replace("Thought: [your reasoning in natural language]", "")
        input_text = llm_model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = llm_model.tokenizer(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        output_text = llm_model.chat(inputs)
        # ic(output_text)
        output_text = output_text[0]
        thought = output_text.split("Thought:")[-1].split("Answer:")[0].strip()
        prediction = output_text.split("Answer:")[-1].strip()
        if "benign" in prediction.lower():
            prediction = 0
        elif "hateful" in prediction.lower():
            prediction = 1
        else:
            logger.error(f"Invalid prediction: {prediction}")
            prediction = 0
        save_df = pd.concat(
            [
                save_df,
                pd.DataFrame(
                    {"vid": vid, "label": label, "explanation": thought, "prediction": prediction},
                    index=[0],
                ),
            ],
            ignore_index=True,
        )
        save_df.to_json(save_path, orient="records", lines=True)
        acc = accuracy_score(save_df["label"].values.astype(int), save_df["prediction"].values.astype(int))
        f1 = f1_score(
            save_df["label"].values.astype(int), save_df["prediction"].values.astype(int), average="macro"
        )
        bar.set_description(f"vid: {vid}, acc: {acc:.4f}, f1: {f1:.4f}")
    acc = accuracy_score(save_df["label"].values.astype(int), save_df["prediction"].values.astype(int))
    f1 = f1_score(
        save_df["label"].values.astype(int), save_df["prediction"].values.astype(int), average="macro"
    )
    print(f"{result_path}. ACC: {acc:.4f}, M-F1: {f1:.4f}")
    return {
        "acc": acc,
        "f1": f1,
    }
