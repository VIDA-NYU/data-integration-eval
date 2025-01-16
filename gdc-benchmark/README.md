# GDC dataset downloading script

This script downloads Excel files from specified URLs, extracts data from specific sheets, and converts them to CSV format. 

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

<!-- ### requirements.txt
```
json
pandas>=1.5.0
requests>=2.28.0
openpyxl>=3.1.0  # Required for Excel file handling
``` -->

## Project Structure

```
.
├── data/
│   ├── downloads/      # Raw downloaded files
│   └── input-tables/   # Processed CSV files
├── papers_info.json    # Dataset configuration file
├── requirements.txt
└── process_data.py     # Main script
```

## Configuration

The script expects a `papers_info.json` file with the following structure:

```json
{
    "paper_name.csv": {
        "Dataset URL": "https://example.com/dataset.xlsx",
        "Sheet Name": "Sheet1"
    }
}
```

## Usage

1. Create the `papers_info.json` file with your dataset configurations (Or use the one provided)
2. Run the script:
   ```bash
   python process_data.py
   ```

<!-- ## Features

- Automatic creation of necessary directories
- Downloads Excel files from specified URLs
- Converts Excel sheets to CSV format
- Special handling of exceptions (removes incompatible rows of Vasaikar dataset)
- Error handling for file downloads and Excel processing -->


## Output

- Downloaded files are saved in `data/downloads/`
- Processed CSV files are saved in `data/input-tables/`


## Notes

- Ensure you have proper permissions to create directories and files
- Check your internet connection for downloading datasets
- Verify the Excel sheet names in your configuration match the source files