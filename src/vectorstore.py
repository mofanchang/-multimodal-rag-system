import json
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStoreManager:
    def __init__(self, collection_name='enterprise_docs', embedding_model='paraphrase-multilingual-mpnet-base-v2'):
        self.embedder = SentenceTransformer(embedding_model, device='cpu')
        self.chroma_client = chromadb.Client()
        self.collection_name = collection_name
        
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={'hnsw:space': 'cosine'}
        )

    def populate_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_docs = data.get('text_docs', []) + data.get('image_docs', []) + data.get('audio_docs', [])
        
        texts = [d['text'] for d in all_docs]
        ids = [d['id'] for d in all_docs]
        metadatas = [{
            'source_id': d['source_id'],
            'modality': d['modality'],
            'category': d['category'],
            'topic': d['topic']
        } for d in all_docs]

        if not texts:
            return 0

        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return self.collection.count()

    def query(self, question, n_results=5, filters=None):
        where = None
        if filters:
            conditions = []
            for k, v in filters.items():
                if v:
                    conditions.append({k: {'$eq': v}})
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {'$and': conditions}

        query_emb = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            where=where
        )
        return results
