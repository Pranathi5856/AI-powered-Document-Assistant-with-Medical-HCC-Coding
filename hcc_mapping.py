import pandas as pd
import re

# Load HCC codes from CSV
def load_hcc_codes(csv_path="resources/hcc_codes.csv"):
    return pd.read_csv(csv_path)

# Simple diagnosis extraction (can be improved with NLP/LLM)
def extract_diagnoses(text):
    diagnoses = re.findall(r'\b(?:diabetes|hypertension|COPD|asthma|CHF|CKD|MI)\b', text, flags=re.IGNORECASE)
    return list(set([d.lower() for d in diagnoses]))

def map_to_hcc_codes(diagnoses, hcc_df):
    mappings = []
    for diag in diagnoses:
        row = hcc_df[hcc_df['Diagnosis'].str.lower() == diag]
        if not row.empty:
            mappings.append({"diagnosis": diag, "hcc_code": row.iloc[0]['HCC_Code'], "description": row.iloc[0]['Description']})
    return mappings