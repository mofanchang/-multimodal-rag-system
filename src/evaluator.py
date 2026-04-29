import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score

class RAGEvaluator:
    def __init__(self, engine):
        self.engine = engine

    def run_evaluation(self, eval_dataset):
        retrieval_hits = []
        reciprocal_ranks = []
        retrieval_precisions = []
        rouge_l_scores = []
        faithfulness_scores = []
        all_preds = []
        all_refs = []

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

        for item in eval_dataset:
            # 1. Retrieval Evaluation
            results = self.engine.vectorstore.query(item['question'], n_results=5)
            retrieved_ids = [m['source_id'] for m in results['metadatas'][0]] if results['metadatas'] else []

            hit = 1 if item['expected_source'] in retrieved_ids else 0
            retrieval_hits.append(hit)

            if hit:
                rank = retrieved_ids.index(item['expected_source']) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            precision = retrieved_ids.count(item['expected_source']) / len(retrieved_ids) if retrieved_ids else 0
            retrieval_precisions.append(precision)

            # 2. Generation Evaluation
            pred_answer, _ = self.engine.query(item['question'])
            all_preds.append(pred_answer)
            all_refs.append(item['reference_answer'])

            rouge_l = scorer.score(item['reference_answer'], pred_answer)['rougeL'].fmeasure
            rouge_l_scores.append(rouge_l)

            # Faithfulness
            pred_emb = self.engine.vectorstore.embedder.encode([pred_answer])
            ref_emb = self.engine.vectorstore.embedder.encode([item['reference_answer']])
            faith_sim = np.dot(pred_emb, ref_emb.T) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
            faithfulness_scores.append(float(faith_sim.item()))

        # BERTScore
        _, _, F1 = bert_score(all_preds, all_refs, lang='zh', verbose=False)

        return {
            "retrieval": {
                "Hit Rate@5": np.mean(retrieval_hits),
                "MRR@5": np.mean(reciprocal_ranks),
                "Precision@5": np.mean(retrieval_precisions)
            },
            "generation": {
                "ROUGE-L": np.mean(rouge_l_scores),
                "BERTScore_F1": F1.mean().item(),
                "Faithfulness": np.mean(faithfulness_scores)
            }
        }
