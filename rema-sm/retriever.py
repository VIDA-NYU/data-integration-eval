import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize

from utils import get_samples

lm_map = {
    "roberta": "roberta-base",
    "mpnet": "microsoft/mpnet-base",
    "arctic": "Snowflake/snowflake-arctic-embed-m-v1.5",
}

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class ColumnRetriever:
    def __init__(self, model_type, dataset, serialization):
        self.serialization = serialization
        self.model_type = model_type
        self._model = self._load_model(model_type, dataset)
        self._tokenizer = AutoTokenizer.from_pretrained(
            lm_map[model_type.split("-")[0]]
        )

    def _load_model(self, model_type, dataset):
        model_key = model_type.split("-")[0]

        if "ft" in model_type:
            model_path = f"{model_key}-{dataset}-{self.serialization}-ft"
        else:
            model_path = lm_map[model_key]

        if "arctic" in model_key:
            model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)
            model.eval()
            return model
        else:
            return AutoModel.from_pretrained(model_path)

    def encode_columns(self, dataframe):
        return {
            col: self._encode_column(col, dataframe[col]) for col in dataframe.columns
        }

    def _encode_column(self, header, values):
        text = self._tokenize(header, values)
        inputs = self._tokenizer(text, return_tensors="pt")
        outputs = self._model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    def _tokenize(self, header, values):
        if self.serialization == "header":
            text = header
        else:
            tokens = get_samples(values)
            text = (
                self._tokenizer.cls_token
                + header
                + self._tokenizer.sep_token
                + self._tokenizer.sep_token.join(tokens)
            )
        return text

    def find_matches(self, source_table, target_table, top_k):
        if "arctic" in self.model_type:
            return self._match_columns_arctic(source_table, target_table, top_k)
        else:
            source_embeddings = self.encode_columns(source_table)
            target_embeddings = self.encode_columns(target_table)
            return self._match_columns(source_embeddings, target_embeddings, top_k)

    def _match_columns(self, source_embeddings, target_embeddings, top_k):
        matched_columns = {}
        for s_col, s_emb in source_embeddings.items():
            similarities = {
                t_col: self._cosine_similarity(s_emb, t_emb)
                for t_col, t_emb in target_embeddings.items()
            }
            sorted_similarities = sorted(
                similarities.items(), key=lambda x: x[1], reverse=True
            )
            matched_columns[s_col] = sorted_similarities[:top_k]

        return matched_columns

    def _cosine_similarity(self, vec1, vec2):
        sim = np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return sim[0][0]

    def _match_columns_arctic(self, source_table, target_table, top_k):
        queries = []
        for col in source_table.columns:
            queries.append(self._tokenize(col, source_table[col]))
        queries_with_prefix = [f"{QUERY_PREFIX}{q}" for q in queries]
        query_tokens = self._tokenizer(
            queries_with_prefix,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        documents = []
        for col in target_table.columns:
            documents.append(self._tokenize(col, target_table[col]))
        document_tokens = self._tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        # Use the model to generate text embeddings.
        with torch.inference_mode():
            query_embeddings = self._model(**query_tokens)[0][:, 0]
            document_embeddings = self._model(**document_tokens)[0][:, 0]

        query_embeddings = normalize(query_embeddings)
        document_embeddings = normalize(document_embeddings)

        scores = query_embeddings @ document_embeddings.T

        # for query, query_scores in zip(queries, scores):
        #     doc_score_pairs = list(zip(documents, query_scores))
        #     doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        #     print(f'Query: "{query}"')
        #     for document, score in doc_score_pairs:
        #         print(f'Score: {score:.4f} | Document: "{document}"')
        #     print()

        matched_columns = {}

        for col, query_scores in zip(source_table.columns, scores):
            doc_score_pairs = list(zip(target_table.columns, query_scores))
            doc_score_pairs = [(doc, score.item()) for doc, score in doc_score_pairs]
            doc_score_pairs_sorted = sorted(
                doc_score_pairs, key=lambda x: x[1], reverse=True
            )
            matched_columns[col] = doc_score_pairs_sorted[:top_k]

        return matched_columns
