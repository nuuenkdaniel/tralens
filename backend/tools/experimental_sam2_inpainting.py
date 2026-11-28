"""Optional SAM 2 / SimpleLaMa prototype. Supply the checkpoint locally."""
from PIL import Image, ImageDraw, ImageFont
from tralens.ocr.types import Text_Group
from simple_lama_inpainting import SimpleLama
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

_sam2_predictor = None
_lama_model = None

def get_sam2_predictor():
    """Lazily load SAM2 to prevent reloading on every function call."""
    global _sam2_predictor
    if _sam2_predictor is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
    return _sam2_predictor

def get_lama_model():
    """Lazily load SimpleLama to prevent reloading on every function call."""
    global _lama_model
    if _lama_model is None:
        _lama_model = SimpleLama()
    return _lama_model

def get_sam2_masks(image: Image.Image, groups: list[Text_Group], padding=5) -> np.ndarray:
    """Helper function to run SAM2 prediction for all bounding boxes at once."""
    boxes = []
    for group in groups:
        for box in group.group:
            x1, y1, x2, y2 = map(int, box.bbox)
            # Add padding to the prompt box to ensure it fully envelopes the text
            x1, y1 = max(0, x1-padding), max(0, y1-padding)
            x2, y2 = min(image.width, x2+padding), min(image.height, y2+padding)
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        # Return a completely black boolean mask if no text is found
        return np.zeros((image.height, image.width), dtype=bool)

    image_np = np.array(image.convert("RGB"))

    predictor = get_sam2_predictor()
    predictor.set_image(image_np)

    # Predict all masks simultaneously. Shape returned is (N, 1, H, W)
    masks, _, _ = predictor.predict(box=np.array(boxes), multimask_output=False)

    # Flatten the (N, 1, H, W) array into a single 2D boolean mask (H, W)
    return np.any(masks.reshape(-1, image.height, image.width), axis=0)

def cover_text(image: Image.Image, groups: list[Text_Group], padding=5) -> Image.Image:
    """Replaces text with exact white segmentation masks instead of rectangles."""
    combined_mask = get_sam2_masks(image, groups, padding)

    new_image_np = np.array(image.convert("RGB"))
    # Apply pure white wherever the SAM2 mask is true
    new_image_np[combined_mask] = [255, 255, 255]

    return Image.fromarray(new_image_np)

def inpaint(image: Image.Image, groups: list[Text_Group], padding=5) -> Image.Image:
    """Uses SAM2 segmentation masks to accurately inpaint only the text."""
    combined_mask = get_sam2_masks(image, groups, padding)

    # Convert the boolean numpy mask to an 8-bit 'L' PIL Image for LaMa
    mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8), mode="L")

    lama = get_lama_model()
    return lama(image.convert("RGB"), mask_image)

def draw_text(image: Image.Image, groups: list[Text_Group], font_path="/usr/share/fonts/liberation/LiberationMono-Regular.ttf"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 12)

    for group in groups:
        for box in group.group:
            # Uncommented to handle Surya's `<br>` line break tags seamlessly during rendering
            text = box.text.replace("<br>", "\n")
            draw.text((box.bbox[0], box.bbox[1]), text, font=font, fill="black")

    return image
