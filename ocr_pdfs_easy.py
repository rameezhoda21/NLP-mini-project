import csv
import re
from pathlib import Path
import fitz  # PyMuPDF
import easyocr
import numpy as np

# ==========================================
# 1. Setup Paths
# ==========================================
BASE_DIR = Path("data")
RAW_PDF_DIR = BASE_DIR / "raw_pdfs"
EXTRACTED_DIR = BASE_DIR / "extracted_text"
CLEANED_DIR = BASE_DIR / "cleaned_text"
CSV_REPORT_PATH = BASE_DIR / "extraction_report.csv"

def clean_text(text: str) -> str:
    """Same cleaning logic used previously."""
    if not text: return ""
    cleaned = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def process_ocr():
    print("Starting Pure-Python OCR process...")
    
    if not CSV_REPORT_PATH.exists():
        print(f"Cannot find {CSV_REPORT_PATH}. Run extraction first.")
        return

    # Find files that failed normal extraction
    needs_ocr_files = []
    with open(CSV_REPORT_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("likely_needs_ocr") == "True":
                needs_ocr_files.append(row["pdf_filename"])

    if not needs_ocr_files:
        print("No files were flagged for OCR in the report.")
        return
        
    print(f"Found {len(needs_ocr_files)} files requiring OCR: {needs_ocr_files}")

    # Initialize the EasyOCR reader (Downloads model on first run)
    print("Initializing EasyOCR (this may take a moment to load the model)...")
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have an Nvidia GPU setup

    for pdf_filename in needs_ocr_files:
        pdf_path = RAW_PDF_DIR / pdf_filename
        base_name = pdf_path.stem
        
        print(f"\nRunning OCR on: {pdf_filename}...")
        
        try:
            # 1. Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            raw_text = ""
            
            # 2. Iterate through pages
            for page_num in range(len(doc)):
                print(f"  - Processing page {page_num + 1}/{len(doc)}")
                page = doc.load_page(page_num)
                
                # Render page to a high-resolution image (matrix)
                # zoom artificially increases resolution for better OCR accuracy
                zoom = 2.0 
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert PyMuPDF image to a format EasyOCR can read (NumPy array)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                # EasyOCR expects RGB, so if the PDF generated RGBA, strip the Alpha channel
                if pix.n == 4: 
                    img_array = img_array[:, :, :3]
                elif pix.n == 1: # Greyscale to RGB
                    img_array = np.stack((img_array.squeeze(),)*3, axis=-1)
                
                # 3. Read text using EasyOCR
                # detail=0 tells it to just return the strings, not the bounding boxes
                result = reader.readtext(img_array, detail=0)
                page_text = "\n".join(result)
                
                raw_text += page_text + "\n\n"
            
            # 4. Save Raw Text
            raw_text_path = EXTRACTED_DIR / f"{base_name}.txt"
            with open(raw_text_path, "w", encoding="utf-8") as rf:
                rf.write(raw_text)
                
            # 5. Clean and Save Text
            cleaned_text = clean_text(raw_text)
            cleaned_text_path = CLEANED_DIR / f"{base_name}.txt"
            with open(cleaned_text_path, "w", encoding="utf-8") as cf:
                cf.write(cleaned_text)
                
            print(f"  [Success] OCR extracted {len(raw_text)} raw chars and {len(cleaned_text)} cleaned chars.")
            
        except Exception as e:
            print(f"  [Error] Failed to process {pdf_filename}. Details: {e}")

if __name__ == "__main__":
    process_ocr()