import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize

from utils import infer_column_dtype, lm_map

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class ColumnRetriever:
    def __init__(self, model_type, dataset, serialization):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        else:
            model = AutoModel.from_pretrained(model_path)

        model.eval()
        model.to(self.device)
        return model

    def encode_columns(self, table, values):
        return {
            col: self._encode_column(col, table[col], values[col])
            for col in table.columns
        }

    def _encode_column(self, header, values, tokens):
        text = self._tokenize(header, values, tokens)
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # Move to CPU

    def _tokenize(self, header, values, tokens):
        if self.serialization == "header":
            return header

        data_type = infer_column_dtype(values)

        if self.serialization == "header_values_default":
            text = (
                self._tokenizer.cls_token
                + header
                + self._tokenizer.sep_token
                + data_type
                + self._tokenizer.sep_token
                + self._tokenizer.sep_token.join(tokens)
            )

        elif self.serialization == "header_values_prefix":
            text = (
                self._tokenizer.cls_token
                + "header:"
                + header
                + self._tokenizer.sep_token
                + " datatype:"
                + data_type
                + self._tokenizer.sep_token
                + " values:"
                + ", ".join(tokens)
            )

        elif self.serialization == "header_values_repeat":
            text = (
                self._tokenizer.cls_token
                + self._tokenizer.sep_token.join([header] * 5)
                + self._tokenizer.sep_token
                + data_type
                + self._tokenizer.sep_token
                + self._tokenizer.sep_token.join(tokens)
            )

        return text

    def find_matches(
        self, source_table, target_table, source_values, target_values, top_k
    ):
        if "arctic" in self.model_type:
            return self._match_columns_arctic(
                source_table, target_table, source_values, target_values, top_k
            )
        else:
            source_embeddings = self.encode_columns(source_table, source_values)
            target_embeddings = self.encode_columns(target_table, target_values)
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

    def _match_columns_arctic(
        self, source_table, target_table, source_values, target_values, top_k
    ):
        queries = []
        for col in source_table.columns:
            queries.append(self._tokenize(col, source_table[col], source_values[col]))
        queries_with_prefix = [f"{QUERY_PREFIX}{q}" for q in queries]
        query_tokens = self._tokenizer(
            queries_with_prefix,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        documents = []
        for col in target_table.columns:
            documents.append(self._tokenize(col, target_table[col], target_values[col]))
        document_tokens = self._tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():
            query_embeddings = self._model(**query_tokens)[0][:, 0]
            document_embeddings = self._model(**document_tokens)[0][:, 0]

        query_embeddings = normalize(query_embeddings)
        document_embeddings = normalize(document_embeddings)

        scores = query_embeddings @ document_embeddings.T
        matched_columns = {}

        for col, query_scores in zip(source_table.columns, scores):
            doc_score_pairs = list(zip(target_table.columns, query_scores))
            doc_score_pairs = [(doc, score.item()) for doc, score in doc_score_pairs]
            doc_score_pairs_sorted = sorted(
                doc_score_pairs, key=lambda x: x[1], reverse=True
            )
            matched_columns[col] = doc_score_pairs_sorted[:top_k]

        return matched_columns
