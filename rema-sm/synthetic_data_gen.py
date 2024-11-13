import argparse
from openai import OpenAI
import json
import random
import os
import sys
import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEY


class SemanticGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def _generate_prompt(self, column_name, column_values):
        if len(column_values) > 0:
            prompt = f"Given the table column '{column_name}' with values {column_values}, \
generate three alternative column names that adhere to typical database naming conventions such as underscores and abbreviations. \
Additionally, provide distinct, technically correct synonyms or variants for the listed values \
For columns with numerical or datetime data, generate random numbers or dates appropriate to the column's semantic meaning. \
Ensure that each set does not exceed 15 values. \
Format your output as follows: \
alternative_name_1, value1, value2, value3, ...; alternative_name_2, value1, value2, value3, ...; alternative_name_3, value1, value2, value3, ... \
Ensure your response excludes additional information and quotations."
        else:
            prompt = f"Given the table column '{column_name}', generate three alternative column names that adhere to typical database naming conventions such as underscores and abbreviations. \
Additionally, suggest distinct, technically accurate values appropriate for the data type of the column. \
Ensure that each set does not exceed 15 values. \
Format your output as follows: \
alternative_name_1, value1, value2, value3, ...; alternative_name_2, value1, value2, value3, ...; alternative_name_3, value1, value2, value3, ... \
Ensure your response excludes additional information and quotations."
        return prompt

    def get_semantic_matches(
        self, column_name, column_values, model="gpt-4-turbo-preview"
    ):
        # print(f"Generating semantic matches for column: {column_name}")
        # print(f"Column values: {column_values}")
        prompt = self._generate_prompt(column_name, column_values)
        messages = [
            {
                "role": "system",
                "content": "You are an AI trained to perform schema matching by providing similarity scores.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        matches_content = response.choices[0].message.content
        matches = matches_content.split("; ")
        alternative_name_values = {}
        for match in matches:
            match = match.replace("\n", "")
            # print(match)
            alternative_name, values = match.split(", ", 1)
            values = values.split(", ")
            alternative_name_values[alternative_name] = values
        return alternative_name_values


class ExactGenerator:
    def __init__(self, threshold=1):
        self.threshold = threshold

    def get_exact_matches(self, column_name, column_values):
        value_size = len(column_values)
        # with probability(0.3) replace a random character in the column name with a random character or number or blank space
        if random.random() < 0.3 or value_size < self.threshold:
            alternative_column_name = list(column_name)
            alternative_column_name[random.randint(0, len(alternative_column_name) - 1)] = random.choice(
                "abcdefghijklmnopqrstuvwxyz0123456789 "
            )
            alternative_column_name = "".join(alternative_column_name)
        else:
            alternative_column_name = column_name
        if value_size < self.threshold:
            return {
                f"{column_name}_1": [],
                f"{alternative_column_name}_2": [],
                # f"{column_name}_3": [],
            }
        return {
            f"{column_name}_1": random.sample(
                column_values, random.randint(1, min(value_size, 15))
            ),
            f"{alternative_column_name}_2": random.sample(
                column_values, random.randint(1, min(value_size, 15))
            ),
            # f"{column_name}_3": random.sample(
            #     column_values, random.randint(1, value_size)
            # ),
        }


def generate_matches(dataset, unique_columns):
    matches = {}
    file_path = f"{dataset}_synthetic_matches.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            matches = json.load(file)

    exact_generator = ExactGenerator()
    semantic_generator = SemanticGenerator(API_KEY)

    for column_name, column_values in tqdm.tqdm(unique_columns.items()):
        if column_name in matches:
            continue

        matches[column_name] = {"exact": {}, "semantic": {}, "original": {}}
        
        values = column_values if len(column_values) < 15 else random.sample(column_values, 15)
        matches[column_name]["original"] = {column_name: values}

        exact_matches = exact_generator.get_exact_matches(column_name, column_values)
        if exact_matches:
            matches[column_name]["exact"].update(exact_matches)

        while True:
            semantic_matches = semantic_generator.get_semantic_matches(
                column_name, column_values
            )
            if len(semantic_matches) == 3:
                break
        if semantic_matches:
            matches[column_name]["semantic"].update(semantic_matches)

        with open(file_path, "w") as file:
            json.dump(matches, file, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Match columns between source and target tables using pretrained models."
    )
    parser.add_argument(
        "--dataset",
        default="gdc",
        help="Name of the dataset",
    )
    args = parser.parse_args()
    dataset = args.dataset

    try:
        with open(f"{dataset}_unique_columns.json", "r") as f:
            unique_columns = json.load(f)
    except FileNotFoundError:
        print(f"Error: {dataset}_unique_columns.json not found.")
        exit()

    generate_matches(dataset, unique_columns)


if __name__ == "__main__":
    main()
