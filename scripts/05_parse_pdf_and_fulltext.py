import os
import requests
from pypdf import PdfReader
import pandas as pd
import xml.etree.ElementTree as ET

raw_dir = os.path.expanduser("~/CalibraNeo_Project/data_raw/zaghla_2026")
pdf_path = os.path.join(raw_dir, "jitc-14-4-s002.pdf")
xlsb_path = os.path.join(raw_dir, "jitc-14-4-s003.xlsb")

print("=== 1. FULL TEXT OF s002.pdf ===")
reader = PdfReader(pdf_path)
pdf_lines = []
for p_idx, page in enumerate(reader.pages):
    lines = [line.strip() for line in page.extract_text().split("\n") if line.strip()]
    print(f"Page {p_idx+1} has {len(lines)} lines.")
    for l in lines:
        pdf_lines.append(l)

print("\nFirst 15 lines of s002.pdf:")
for l in pdf_lines[:15]:
    print(" ", l)
print("\nLast 10 lines of s002.pdf:")
for l in pdf_lines[-10:]:
    print(" ", l)

print("\n=== 2. CHECKING EUROPE PMC FULL-TEXT XML FOR EMBEDDED TABLES ===")
xml_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13059925/fullTextXML"
resp = requests.get(xml_url, timeout=30)
if resp.status_code == 200:
    xml_path = os.path.join(raw_dir, "PMC13059925_fulltext.xml")
    with open(xml_path, "wb") as f:
        f.write(resp.content)
    root = ET.fromstring(resp.content)
    tables = root.findall(".//table-wrap")
    print(f"Found {len(tables)} tables in main article XML.")
    for t in tables:
        label = t.findtext(".//label")
        caption = "".join(t.find(".//caption").itertext()).strip() if t.find(".//caption") is not None else "No caption"
        print(f"  - {label}: {caption[:120]}...")
        
    # Also check supplementary-material links inside XML
    supps = root.findall(".//supplementary-material")
    print(f"Found {len(supps)} supplementary-material tags in XML:")
    for s in supps:
        s_text = " ".join("".join(s.itertext()).split())
        print(f"  * {s.attrib} -> {s_text[:120]}")

print("\n=== 3. CHECKING MUTATED PROTEIN NAMES IN s003.xlsb ===")
df = pd.read_excel(xlsb_path, sheet_name="CLean_Filtered Data", engine="pyxlsb")
# Filter ProteinName that does NOT start with standard human sp| or tr| (or check unique patterns)
prot_names = df["ProteinName"].dropna().unique()
print(f"Total unique ProteinName entries in s003.xlsb: {len(prot_names)}")
mutant_prots = [p for p in prot_names if "_" in str(p) and not str(p).startswith((">sp|", ">tr|", ">Reverse", ">contaminant"))]
print(f"Detected {len(mutant_prots)} custom/mutant protein headers! First 20:")
for mp in mutant_prots[:20]:
    print("  ", mp)
