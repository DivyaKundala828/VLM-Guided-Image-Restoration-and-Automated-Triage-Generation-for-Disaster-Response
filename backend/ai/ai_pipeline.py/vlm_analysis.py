import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def analyze_image(image_path):

    print("Loading SmolVLM...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    else:

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        )

        model = model.to(device)

    print("SmolVLM loaded successfully.")

    image = Image.open(image_path).convert("RGB")

    prompt_template = """
You are a disaster-response visual analysis assistant.

Analyze the provided disaster image and answer the following questions
using only information that can be observed in the image.

Return your answer in this exact format:

Disaster type: <type>
People visible: <Yes/No/Unclear>
Infrastructure damage: <Yes/No/Unclear>
Road passable: <Yes/No/Unclear>
Major hazards: <short description>

Questions:
1. What type of disaster is visible in the image?
2. Are there people visible in the image?
3. Is there visible damage to infrastructure?
4. Is the road or pathway passable?
5. What major hazards are visible in the image?
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": prompt_template
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }

    print("Running disaster analysis...")

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100
        )

    input_length = inputs["input_ids"].shape[1]

    generated_answer = generated_ids[:, input_length:]

    report = processor.batch_decode(
        generated_answer,
        skip_special_tokens=True
    )[0]

    print("===== DISASTER RESPONSE VISUAL REPORT =====")
    print(report)

    return report


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python vlm_analysis.py <image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    analyze_image(image_path)