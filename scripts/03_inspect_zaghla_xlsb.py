import os
import pandas as pd
import docx

raw_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/zaghla_2026")
xlsb_path = os.path.join(raw_dir, "jitc-14-4-s003.xlsb")
docx_path = os.path.join(raw_dir, "jitc-14-4-s001.docx")

print("=== 1. READING WORD DOC SUPPLEMENTARY NOTES (s001.docx) ===")
if os.path.exists(docx_path):
    doc = docx.Document(docx_path)
    for p in doc.paragraphs:
        if p.text.strip():
            print(" ", p.text.strip()[:200])

print("\n=== 2. INSPECTING ALL SHEETS IN BINARY EXCEL (s003.xlsb) ===")
xl = pd.ExcelFile(xlsb_path, engine="pyxlsb")
print(f"Total Sheets Found: {len(xl.sheet_names)}")
print("Sheet Names:", xl.sheet_names)

for sheet in xl.sheet_names:
    print(f"\n--- SHEET: [{sheet}] ---")
    df = pd.read_excel(xlsb_path, sheet_name=sheet, engine="pyxlsb")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("First 10 Columns:", list(df.columns)[:10])
    print("Preview (first 2 rows):")
    print(df.head(2).to_string())
