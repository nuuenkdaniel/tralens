from ollama import generate
from ollama import GenerateResponse

target_language = "English"
target_code = "en"

def translate(text):
    context = f"""
    You are a professional any language to {target_language} ({target_code}) translator. Your goal is to accurately convey the meaning and nuances of the original text while adhering to {target_language} grammar, vocabulary, and cultural sensitivities.
Produce only the {target_language} translation, without any additional explanations or commentary. Please translate the following text into {target_language}:

{text}
    """
    response = generate(model="translategemma:27b", prompt=context)
    return response["response"]
