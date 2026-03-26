import csv
import re
import PyPDF2
from pathlib import Path

# ==========================================
# 1. Setup Paths and Directories
# ==========================================
# We use pathlib for robust, cross-platform file path handling
BASE_DIR = Path("data")
RAW_PDF_DIR = BASE_DIR / "raw_pdfs"
EXTRACTED_DIR = BASE_DIR / "extracted_text"
CLEANED_DIR = BASE_DIR / "cleaned_text"

# Create the required directories if they don't exist
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from a PDF based on specific requirements:
    - Removes isolated page numbers
    - Removes repeated spaces
    - Removes excessive blank lines
    - Normalizes line breaks
    """
    if not text:
        return ""

    # Remove isolated page numbers (lines that contain only numbers and optional whitespace)
    # (?m) enables multiline mode so ^ and $ match the start/end of every line
    cleaned = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    
    # Remove repeated spaces (2 or more spaces become a single space)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    # Normalize line breaks: Strip trailing/leading spaces from each line
    lines = [line.strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)
    
    # Remove excessive blank lines (more than 2 consecutive newlines become just 2)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

def process_pdfs():
    """Reads PDFs, extracts text, cleans it, saves both versions, and prints a summary."""
    
    print("Starting PDF extraction process...")
    summary_report = []
    
    # Find all PDF files in the raw_pdfs folder
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {RAW_PDF_DIR}. Please place your PDFs there and run again.")
        return

    # Process each PDF file
    for pdf_path in pdf_files:
        filename = pdf_path.name
        base_name = pdf_path.stem
        
        raw_text = ""
        extraction_success = False
        needs_ocr = False
        
        print(f"Processing: {filename}")
        
        # 2. Extract text from the PDF
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                # Loop through all pages and extract text
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        raw_text += page_text + "\n"
            
            extraction_success = True
            
        except Exception as e:
            print(f"  [!] Failed to extract {filename}. Error: {e}")
            
        # 3. Analyze extracted text quality (Character count & OCR flagging)
        raw_char_count = len(raw_text)
        
        # If extraction "succeeded" but we got very little text, it's likely a scanned image
        # Standard threshold: less than 100 characters combined usually means it's a scanned page
        if extraction_success and raw_char_count < 100:
            needs_ocr = True
            
        # 4. Save Raw Text
        raw_text_path = EXTRACTED_DIR / f"{base_name}.txt"
        with open(raw_text_path, "w", encoding="utf-8") as raw_file:
            raw_file.write(raw_text)
            
        # 5. Clean Text
        cleaned_text = clean_text(raw_text)
        clean_char_count = len(cleaned_text)
        
        # 6. Save Cleaned Text
        cleaned_text_path = CLEANED_DIR / f"{base_name}.txt"
        with open(cleaned_text_path, "w", encoding="utf-8") as clean_file:
            clean_file.write(cleaned_text)
            
        # Append data to our summary list
        summary_report.append({
            "pdf_filename": filename,
            "raw_txt_filename": f"{base_name}.txt",
            "cleaned_txt_filename": f"{base_name}.txt",
            "extraction_success": extraction_success,
            "raw_char_count": raw_char_count,
            "cleaned_char_count": clean_char_count,
            "likely_needs_ocr": needs_ocr
        })
        
    # ==========================================
    # 7. Print Terminal Summary
    # ==========================================
    print("\n" + "="*85)
    print("EXTRACTION SUMMARY REPORT")
    print("="*85)
    
    # Format a table header
    header = f"{'Filename':<35} | {'Success':<8} | {'Raw Chars':<10} | {'Clean Chars':<11} | {'Needs OCR'}"
    print(header)
    print("-" * 85)
    
    # Print each document's results
    for doc in summary_report:
        # Truncate filename if it's too long for the table column
        fname_display = doc['pdf_filename'][:32] + "..." if len(doc['pdf_filename']) > 35 else doc['pdf_filename']
        
        row = (f"{fname_display:<35} | "
               f"{str(doc['extraction_success']):<8} | "
               f"{doc['raw_char_count']:<10} | "
               f"{doc['cleaned_char_count']:<11} | "
               f"{doc['likely_needs_ocr']}")
        print(row)
        
    print("="*85)
    
    # ==========================================
    # 8. Save CSV Report
    # ==========================================
    csv_path = BASE_DIR / "extraction_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "pdf_filename", "raw_txt_filename", "cleaned_txt_filename",
            "extraction_success", "raw_char_count", "cleaned_char_count",
            "likely_needs_ocr"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_report)

    print(f"Processing complete! Text files saved in {EXTRACTED_DIR} and {CLEANED_DIR}")
    print(f"Extraction report saved at: {csv_path}")

if __name__ == "__main__":
    process_pdfs()
