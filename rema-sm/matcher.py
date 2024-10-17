from openai import OpenAI
import tiktoken

from utils import get_samples


class ColumnMatcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def num_tokens_from_string(self, string, encoding_name="gpt-4-turbo-preview"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def rematch(self, source_table, matched_columns, top_k, score_based=True):
        refined_matches = {}
        for source_col, target_col_scores in matched_columns.items():
            target_cols = [target_col for target_col, score in target_col_scores]
            targets = "; ".join(target_cols)
            tokens = get_samples(source_table[source_col])
            cand = source_col + " " + " ".join(tokens)
            if score_based:
                refined_match = self._get_matches_w_score(cand, targets, top_k)
                refined_match = self._parse_scored_matches(refined_match)
            else:
                refined_match = self._get_matches(cand, targets, top_k)
                refined_match = refined_match.split("; ")
            refined_matches[source_col] = refined_match
        return refined_matches

    def _get_matches(self, cand, targets, k, model="gpt-4-turbo-preview"):
        messages = [
            {
                "role": "system",
                "content": "You are an assistant for schema matching.",
            },
            {
                "role": "user",
                "content": """ Please select the top """
                + str(k)
                + """ schemas from """
                + targets
                + """ which best matches the candidate column, which is defined by the column name followed by its respective values. Please respond only with the name of the classes separated by semicolon.
                    \n CONTEXT: """
                + cand
                + """ \n RESPONSE: \n""",
            },
        ]
        col_type = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        col_type_content = col_type.choices[0].message.content
        return col_type_content

    def _get_matches_w_score(self, cand, targets, k, model="gpt-4-turbo-preview"):
        messages = [
            {
                "role": "system",
                "content": "You are an AI trained to perform schema matching by providing similarity scores.",
            },
            {
                "role": "user",
                "content": """Please evaluate and score the top """
                + str(k)
                + """schemas from the following list which best match the candidate column. The candidate column is defined by the column name and a sample of its respective values. \
                    Provide only the name of each schema followed by its similarity score (0.00 - 1.00) in parentheses, formatted to two decimals, and separated by a semicolon. Rank the results by score in descending order. \
                    Here's an example of the expected format: "schema1(0.95); schema2(0.73); schema3(0.50); schema4(0.29); schema5(0.04)". \
                    \n\nCandidate Column with values: """
                + cand
                + """\n\nTarget Schemas:"""
                + targets
                + """\n\nResponse:""",
            },
        ]
        col_type = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        col_type_content = col_type.choices[0].message.content
        print(col_type_content)
        return col_type_content

    def _parse_scored_matches(self, refined_match):
        matched_columns = []
        entries = refined_match.split("; ")

        for entry in entries:
            schema_part, score_part = entry.split("(")
            score = float(score_part[:-1])
            schema_name = schema_part.strip()
            matched_columns.append((schema_name, score))

        return matched_columns
