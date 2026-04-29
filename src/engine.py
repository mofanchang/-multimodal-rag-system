import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import CrossEncoder

PROMPT_TEMPLATES = {
    'v1': '你是一個專業的企業助理。你必須只使用台灣繁體中文回答，絕對不可以使用任何簡體字。根據參考資料回答問題。',
    'v2': '你是企業助理。你必須只使用台灣繁體中文。請用以下格式回答：\n【結論】\n【依據】\n【補充】\n絕對不可以使用簡體字。',
    'v3': '你是企業助理。你必須只使用台灣繁體中文，用2-3句話精簡回答。絕對不可以使用簡體字。'
}

class RAGEngine:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.reranker = None
        self.gen_tokenizer = None
        self.gen_model = None
        
    def initialize_models(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        # Reranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Generation Model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )
        
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        self.gen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            kwargs["quantization_config"] = bnb_config
            kwargs["device_map"] = device_map

        self.gen_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def query(self, question, n_results=5, rerank_top_k=3, filter_topic=None, filter_modality=None, prompt_version='v1'):
        filters = {'topic': filter_topic, 'modality': filter_modality}
        results = self.vectorstore.query(question, n_results=n_results, filters=filters)
        
        if not results['documents'] or not results['documents'][0]:
            return "無法檢索到相關文獻。", []

        retrieved_docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        # Rerank
        pairs = [(question, doc) for doc in retrieved_docs]
        rerank_scores = self.reranker.predict(pairs)
        top_idx = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)[:rerank_top_k]

        final_contexts = []
        source_display_list = []
        
        for i in top_idx:
            doc = retrieved_docs[i]
            meta = metas[i]
            final_contexts.append(f"資料來源 [{meta['source_id']}]: {doc}")
            source_display_list.append(meta)

        context_text = "\n\n".join(final_contexts)
        
        # Generate
        messages = [
            {'role': 'system', 'content': PROMPT_TEMPLATES.get(prompt_version, PROMPT_TEMPLATES['v1'])},
            {'role': 'user', 'content': f'參考資料：\n{context_text}\n\n問題：{question}'}
        ]

        device = self.gen_model.device if hasattr(self, 'gen_model') and self.gen_model is not None else "cpu"
        inputs = self.gen_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)

        if self.gen_tokenizer.pad_token_id is None:
            self.gen_tokenizer.pad_token_id = self.gen_tokenizer.eos_token_id

        try:
            with torch.no_grad():
                outputs = self.gen_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.gen_tokenizer.pad_token_id,
                    eos_token_id=self.gen_tokenizer.eos_token_id
                )
            input_len = inputs['input_ids'].shape[1]
            answer = self.gen_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        except Exception as e:
            answer = f"生成錯誤：{str(e)}"
            
        return answer, source_display_list
