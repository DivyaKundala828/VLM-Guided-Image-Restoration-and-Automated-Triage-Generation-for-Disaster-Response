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

    # Load image
    image = Image.open(image_path).convert("RGB")


    # Prompt for disaster classification
    prompt_template = """
You are an emergency disaster image classification system.

Look carefully at the image and classify the PRIMARY disaster
that is visibly happening.

You MUST choose exactly ONE disaster type from this list:

Fire
Flood
Earthquake
Landslide
Storm
Hurricane
Accident
Wildfire
Other

IMPORTANT CLASSIFICATION RULES:

1. FIRE:
If flames, a burning building, active fire, or strong evidence
of fire is clearly visible, choose Fire.

2. FLOOD:
Choose Flood ONLY when substantial water or flooding is clearly
visible in the image.

Do NOT choose Flood because of:
- a damaged road
- a damaged building
- smoke
- fire
- a muddy area without clear flooding

3. EARTHQUAKE:
Choose Earthquake only when earthquake-related destruction
is visually evident.

4. LANDSLIDE:
Choose Landslide only when soil, rocks, or earth have visibly
moved over an area.

5. STORM:
Choose Storm only when storm-related effects are clearly visible.

6. HURRICANE:
Choose Hurricane only when hurricane-related effects are
clearly visible.

7. ACCIDENT:
Choose Accident when a vehicle crash or similar accident
is the main visible event.

8. WILDFIRE:
Choose Wildfire when vegetation, forest, or grassland is
actively burning.

9. OTHER:
Choose Other if none of the above can be confidently
identified from the image.

VERY IMPORTANT:

If flames are clearly visible in a building, choose Fire.

DO NOT classify a burning building as Flood.

DO NOT choose Flood unless actual flooding or substantial
water is visible.

Use only the visual evidence in the image.
Do not guess.

Return ONLY this exact numbered format:

1. <one disaster type from the list>
2. <Yes/No/Unclear>
3. <Yes/No/Unclear>
4. <Yes/No/Unclear>
5. <short description of visible hazards>

Question meanings:

1. What is the primary disaster visible in the image?
2. Are people visible?
3. Is there visible infrastructure damage?
4. Is the road or pathway passable?
5. What major hazards are visibly present?
"""


    # Create VLM conversation
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": prompt_template
                }
            ]
        }
    ]


    # Convert conversation into model prompt
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )


    # Prepare model inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )


    # Move tensors to GPU/CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }


    print("Running disaster analysis...")


    # Generate answer
    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )


    # Remove input tokens from generated output
    input_length = inputs["input_ids"].shape[1]

    generated_answer = generated_ids[:, input_length:]


    # Convert generated tokens to text
    report = processor.batch_decode(
        generated_answer,
        skip_special_tokens=True
    )[0].strip()


    # Print result for debugging
    print("===== DISASTER RESPONSE VISUAL REPORT =====")
    print(report)
    print("============================================")


    return report


# ---------------------------------------------------
# Main program
# ---------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python vlm_analysis.py <image_path>"
        )

        sys.exit(1)


    image_path = sys.argv[1]

    analyze_image(image_path)
