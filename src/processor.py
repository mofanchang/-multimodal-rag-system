import os
import glob
import json
import gc
import torch
import whisper
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class DataProcessor:
    def __init__(self, data_dir="data/sample_dataset"):
        self.data_dir = data_dir
        self.vlm_model = None
        self.vlm_processor = None
        self.whisper_model = None
        self.meta_map = self._load_metadata()

    def _load_metadata(self):
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return {item['id']: item for item in metadata}

    def process_text(self):
        txt_docs = []
        for path in sorted(glob.glob(os.path.join(self.data_dir, '*.txt'))):
            doc_id = os.path.basename(path).replace('.txt', '')
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            meta = self.meta_map.get(doc_id, {})
            txt_docs.append({
                'id': f'{doc_id}_txt',
                'text': text,
                'source_id': doc_id,
                'modality': 'text',
                'category': meta.get('category', ''),
                'topic': meta.get('topic', '')
            })
        return txt_docs

    def load_vlm(self):
        if self.vlm_model is None:
            model_name = 'Qwen/Qwen2-VL-2B-Instruct'
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
            ).eval()
            self.vlm_processor = AutoProcessor.from_pretrained(model_name)

    def process_images(self):
        self.load_vlm()
        img_docs = []
        for path in sorted(glob.glob(os.path.join(self.data_dir, '*.png'))):
            doc_id = os.path.basename(path).replace('.png', '')
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": "請用台灣繁體中文詳細描述這張圖片的內容，包括文字、圖表、重點資訊，避免使用簡體字。"},
                ],
            }]
            text_prompt = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.vlm_processor(
                text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=300, use_cache=True)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                response = self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            meta = self.meta_map.get(doc_id, {})
            img_docs.append({
                'id': f'{doc_id}_img',
                'text': f'[圖片描述] {response}',
                'source_id': doc_id,
                'modality': 'image',
                'category': meta.get('category', ''),
                'topic': meta.get('topic', '')
            })
            gc.collect()
            torch.cuda.empty_cache()
        return img_docs

    def process_audio(self):
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model('medium')
        audio_docs = []
        for path in sorted(glob.glob(os.path.join(self.data_dir, '*.mp3'))):
            doc_id = os.path.basename(path).replace('.mp3', '')
            result = self.whisper_model.transcribe(path, language='zh')
            meta = self.meta_map.get(doc_id, {})
            audio_docs.append({
                'id': f'{doc_id}_audio',
                'text': f'[音訊轉錄] {result["text"].strip()}',
                'source_id': doc_id,
                'modality': 'audio',
                'category': meta.get('category', ''),
                'topic': meta.get('topic', '')
            })
        return audio_docs

    def run_all(self, output_file="processed_dataset.json"):
        txt = self.process_text()
        img = self.process_images()
        aud = self.process_audio()
        export_data = {'text_docs': txt, 'image_docs': img, 'audio_docs': aud}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)
        return export_data
