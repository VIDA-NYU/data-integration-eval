import json
import os
import requests
import pandas as pd


with open('papers_info.json', 'r') as file:
    papers_info = json.load(file)


data_folder = 'data'
download_folder = os.path.join(data_folder, 'downloads')
prepared_folder = os.path.join(data_folder, 'input-tables')
os.makedirs(download_folder, exist_ok=True)
os.makedirs(prepared_folder, exist_ok=True)


def extract_csv_from_excel(excel_file_name, sheet_name):
    try:
        df = pd.read_excel(excel_file_name, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error extracting sheet {sheet_name} from {
              excel_file_name}: {str(e)}")
        return None


for paper in papers_info:
    dataset_url = papers_info[paper]['Dataset URL']
    if dataset_url:
        response = requests.get(dataset_url)
        if response.status_code == 200:
            download_file_name = os.path.join(
                download_folder, os.path.basename(dataset_url))
            with open(download_file_name, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded {dataset_url} to {download_file_name}")

            if download_file_name.endswith('.xlsx'):
                sheet_name = papers_info[paper]['Sheet Name']

                df = extract_csv_from_excel(download_file_name, sheet_name)

                if paper == "Vasaikar.csv":
                    # remove incompatilhe rows in Vasaiker dataset
                    df = df.iloc[1:-1]

                if df is not None:
                    csv_file_name = os.path.join(prepared_folder, paper)
                    df.to_csv(csv_file_name, index=False)
                    print(f"Created CSV file: {csv_file_name}")
        else:
            print(f"Failed to download {dataset_url}")
