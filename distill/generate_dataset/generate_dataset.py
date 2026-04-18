import json
import os

import pandas as pd
from datasets import Dataset
from icecream import ic
from PIL import Image


def balance_dataset_by_downsampling(df, label_column="label"):
    """
    Balance a dataset by downsampling the majority class.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset
        label_column (str): Name of the column containing the class labels

    Returns:
        pandas.DataFrame: A new DataFrame with balanced classes
    """
    # Count samples in each class
    class_counts = df[label_column].value_counts()

    # Identify minority class and its count
    minority_class = class_counts.idxmin()
    minority_count = class_counts[minority_class]

    # Create a list to store dataframes for each class
    class_dfs = []

    # For each class, either keep as is (if minority) or downsample (if majority)
    for class_label, count in class_counts.items():
        class_df = df[df[label_column] == class_label]

        if count > minority_count:
            # Downsample the majority class to match minority class count
            downsampled_df = class_df.sample(n=minority_count, replace=False, random_state=42)
            class_dfs.append(downsampled_df)
        else:
            # Just add the minority class as is
            class_dfs.append(class_df)

    # Combine all the balanced classes
    balanced_df = pd.concat(class_dfs)

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=2025).reset_index(drop=True)

    return balanced_df


def generate_dataset_reason(dataset_name, split="train", **kargs):
    data_df = pd.read_json(f"data/{dataset_name}/data.jsonl", lines=True)
    ablation = kargs.get("ablation", "")
    limit = kargs.get("limit", 1.0)
    print(f"Ablation: {ablation}")
    if split == "train":
        split_vids = pd.read_csv(f"data/{dataset_name}/vids/train.csv").iloc[:, 0].tolist()
    elif split == "valid":
        split_vids = pd.read_csv(f"data/{dataset_name}/vids/valid.csv").iloc[:, 0].tolist()
    else:
        raise ValueError(f"Invalid split: {split}")
    data_df = data_df[data_df["vid"].isin(split_vids)]
    knowledge_df = pd.read_json(f"result/knowledge/{dataset_name}/knowledge_pure.jsonl", lines=True)
    data_df = data_df.sample(frac=limit, random_state=2025).reset_index(drop=True)
    for index, row in data_df.iterrows():
        vid = row["vid"]
        # ic(vid)
        title = row["title"]
        transcript = row["transcript"][:10000]
        ocr = row["ocr"][:10000]
        label = row["label"]
        image_paths = [f"data/{dataset_name}/frames_16/{vid}/frame_{i:03d}.jpg" for i in range(0, 16)]
        # images = [Image.open(path).convert("RGB") for path in image_paths]
        images = image_paths
        # if "4img" in kargs.get("mark", ""):
        if True:
            # ic("Using 4 images")
            images = [image_paths[0], image_paths[4], image_paths[8], image_paths[12]]

        description = knowledge_df[knowledge_df["vid"] == vid]["description"].values[0]

        input_text = f"""
You are a video content analyzer on a video platform.

You are provided with the following inputs from a video:
- 16 consecutive video frames (visual context),
- the video title,
- the full transcript,
- any on-screen text extracted via OCR.

Video Title: [{title}]  
Video Transcript: [{transcript}]  
On-screen Text: [{ocr}]

Your task is to jointly consider all the inputs and write a detailed and logically complete analysis of the video that clearly conveys the main storyline.  
Try to understand the nuanced sentiment of the content, identify key people or objects, provide relevant background context, and describe important visual elements such as symbols or actions.

Respond in the following format: "The video describes [analysis]".
"""
        output_text = description
        # check if text is empty of image path not exists
        if not output_text or not input_text or not all(os.path.exists(path) for path in images):
            ic(input_text, output_text, images)
            exit()
        yield {"images": images, "input_text": input_text, "output_text": output_text}


def generate_dataset_explain(dataset_name, split="train", use_label=False, **kargs):
    data_df = pd.read_json(f"data/{dataset_name}/data.jsonl", lines=True)
    ablation = kargs.get("ablation", False)
    limit = kargs.get("limit", 1.0)
    if split == "train":
        split_vids = pd.read_csv(f"data/{dataset_name}/vids/train.csv").iloc[:, 0].tolist()
    elif split == "valid":
        split_vids = pd.read_csv(f"data/{dataset_name}/vids/valid.csv").iloc[:, 0].tolist()
    else:
        raise ValueError(f"Invalid split: {split}")
    data_df = data_df[data_df["vid"].isin(split_vids)]
    knowledge_df = pd.read_json(f"result/knowledge/{dataset_name}/knowledge_pure.jsonl", lines=True)
    if ablation == "wo-cot":
        knowledge_df = pd.read_json(
            f"result/knowledge-wo-cot/{dataset_name}/knowledge_pure.jsonl", lines=True
        )
    # shuffle the data
    data_df = data_df.sample(frac=limit, random_state=2025).reset_index(drop=True)
    # data_df = balance_dataset_by_upsampling(data_df)
    if split == "train":
        data_df = balance_dataset_by_downsampling(data_df)
    elif split == "valid":
        data_df = data_df.sample(frac=0.1, random_state=2025).reset_index(drop=True)
    for index, row in data_df.iterrows():
        vid = row["vid"]
        # ic(vid)
        title = row["title"]
        transcript = row["transcript"][:10000]
        ocr = row["ocr"][:10000]
        label = row["label"]
        image_paths = [f"data/{dataset_name}/frames_16/{vid}/frame_{i:03d}.jpg" for i in range(0, 16)]
        # images = [Image.open(path).convert("RGB") for path in image_paths]
        images = image_paths
        if True:
            images = [image_paths[0], image_paths[4], image_paths[8], image_paths[12]]

        if ablation == "wo-ground" or ablation == "wo-cot":
            ic("Setting ablation to wo-ground or wo-cot")
            explanation = knowledge_df[knowledge_df["vid"] == vid]["explanation"].values[0]
        elif use_label:
            explanation = knowledge_df[knowledge_df["vid"] == vid]["fix_label_explanation"].values[0]
        else:
            explanation = knowledge_df[knowledge_df["vid"] == vid]["fix_grounding_explanation"].values[0]

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
        match label:
            case 1:
                text_label = "hateful"
            case 0:
                text_label = "benign"
            case _:
                raise ValueError(f"Invalid label: {label}")
        output_text = f"Thought: {explanation.strip()}\nAnswer: {text_label}"
        if ablation == "wo-label":
            ic("Setting ablation to wo-label")
            input_text = input_text.split("Answer:")[0].strip()
            output_text = output_text.split("Answer:")[0].strip()
        elif ablation == "label-only":
            ic("Setting ablation to label-only")
            input_text = input_text.replace(
                "Provide a clear and easy-to-understand reasoning for your judgment.", ""
            )
            input_text = input_text.replace("Thought: [your reasoning in natural language]", "")
            output_text = f"Answer: {text_label}"
        yield {"images": images, "input_text": input_text, "output_text": output_text}


# reason_dataset = Dataset.from_generator(lambda: generate_dataset_reason("HateMM"))


def get_top_k_longest_samples(dataset, k=10):
    # Store (index, total_length) pairs for sorting
    length_info = []

    # Calculate total length (input + output) for each sample
    for idx, sample in enumerate(dataset):
        total_length = len(sample["input_text"]) + len(sample["output_text"])
        length_info.append((idx, total_length))

    # Sort by length in descending order and get top k
    length_info.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in length_info[:k]]
    top_k_size_info = length_info[:k]

    return top_k_indices, top_k_size_info


def convert_to_conversation(sample):
    images = [Image.open(path).convert("RGB") for path in sample["images"]]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["input_text"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["output_text"]}]},
    ]
    for i, img in enumerate(images):
        conversation[0]["content"].append({"type": "image", "image": img})
    return {"messages": conversation}


if __name__ == "__main__":
    explain_dataset = Dataset.from_generator(lambda: generate_dataset_explain("MHClipEN"))
    print(explain_dataset[0])
