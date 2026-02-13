from PIL import Image, ImageDraw, ImageFont
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from ai import translate

# image = Image.open("/home/Danuu/Downloads/german-road-signs-header.jpg")
image = Image.open("/home/Danuu/Downloads/lmgJZ.jpg")
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()

predictions = recognition_predictor([image], det_predictor=detection_predictor)

draw = ImageDraw.Draw(image)

# def get_font_size(init_font_size, bbox_height):
#     font = ImageFont.truetype("arial.ttf", init_font_size)
#     _, text_height = draw.textsize(text, font=font)
#     return 

for text_box in predictions[0].text_lines:
    bbox = text_box.bbox
    confidence = text_box.confidence
    translated = translate(text_box.text)
    print("{")
    print(f"Confidence: {confidence}")
    print(f"bbox: {bbox}")
    print(f"Text: {text_box.text}")
    print(translated)
    print("},")
    bbox_width = bbox[2]-bbox[0]
    bbox_height = bbox[3]-bbox[1]
    print(bbox_width)
    print(bbox_height)
    init_font_size = 12
    
    font = ImageFont.truetype("LiberationSans-Regular.ttf", init_font_size)
    if confidence != None and confidence > .3:
        draw.rectangle(bbox, fill="white")
        draw.text((bbox[0], bbox[1]), translated)
image.save("output.jpg")
