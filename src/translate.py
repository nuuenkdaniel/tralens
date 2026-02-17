from ollama import Client
from ocr import OCR, Text_Group, Text_Box
import re
import json

class Translate:
    def __init__(self, model: str = "gemma3:27b", ollama_host: str = "127.0.0.1") -> None:
        print("Loading TranslateGemma")
        self.client = Client(host=ollama_host)
        self.client.generate(model=model, prompt="")

    def _sort_group(self, text_group: Text_Group) -> list[Text_Box]:
        group = text_group.group
        total_vertical = 0
        for text_box in group:
            bbox = text_box.bbox
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            if h > w:
                total_vertical += 1

        is_vertical = True if total_vertical > len(group)-total_vertical else False
        if is_vertical:
            return sorted(group, key=lambda b: (-b.bbox[0], b.bbox[1]))
        else:
            return sorted(group, key=lambda b: (b.bbox[1], b.bbox[0]))

    def _extract_json_lists(self, raw_text: str) -> list:
        try:
            match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []

    def translate_groups(self, groups: list[Text_Group], image_path) -> list[Text_Group]:
        translated_groups = []
        for i, group in enumerate(groups):
            sorted_group = self._sort_group(group)
            if not sorted_group: continue
            lines = "\n".join([f"Line {i+1}: {b.text}" for i, b in enumerate(sorted_group)])
            prompt = f"""
                You are a professional translator. 

                Task:
                1. Translate each "Line" below individually into English.
                2. Use the other lines in the group as CONTEXT to determine the correct meaning (e.g., for ambiguous words).
                3. Use the provided image as CONTEXT and error correction
                4. Do NOT merge the lines. Keep them separate.
                5. **Handling Line Breaks (<br>):**
                   - If the text is a **paragraph** or **list**, PRESERVE the `<br>` tags in your translation.
                   - If the text is a **bilingual duplicate** (e.g., Japanese <br> English), output ONLY the English translation (remove the break).
                6. Just translate, do NOT try to explain or give any additional comments

                Input Text Group:
                {lines}

                Output strictly a JSON list of objects (one for each line):
                [
                    {{ "id": 1, "original": "...", "translation": "..." }},
                    {{ "id": 2, "original": "...", "translation": "..." }}
                ]
                """
            response = self.client.chat(
                model="gemma3:4b",
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }]
            )
            translated_json = self._extract_json_lists(response["message"]["content"])

            translated_group = []
            for i, box in enumerate(sorted_group):
                match = next((item for item in translated_json if item.get('id') == i+1), None)
                if match:
                    translation = match.get("translation", "")
                else:
                    translation = "Failed"

                translated_text_box = Text_Box(box.confidence, box.bbox, translation)
                translated_group.append(translated_text_box)

            translated_groups.append(Text_Group(translated_group))
        return translated_groups
