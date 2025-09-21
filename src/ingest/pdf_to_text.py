#!/usr/bin/env python3
# python src/ingest/pdf_to_text.py --in data/raw_pdfs --out data/processed/text.jsonl
"""
PDF to text extraction with header/footer stripping.
Processes Employment Act Malaysia PDFs and amendments.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Install with: pip install PyMuPDF")
    exit(1)


def clean_text(text: str) -> str:
    """Clean extracted text by removing headers, footers, and formatting artifacts."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip common header/footer patterns
        if any(pattern in line.lower() for pattern in [
            'employment act',
            'page ',
            'act ',
            'laws of malaysia',
            'federal government gazette',
            'printed by',
            'published by'
        ]):
            # Only skip if it's a short line (likely header/footer)
            if len(line) < 100:
                continue
        
        # Skip page numbers and short administrative text
        if re.match(r'^\d+$', line) or len(line) < 10:
            continue
            
        # Clean up whitespace and formatting
        line = re.sub(r'\s+', ' ', line)
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_metadata(pdf_path: Path) -> Dict[str, str]:
    """Extract metadata from PDF file."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()
        
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', ''),
            'filename': pdf_path.name,
            'file_size': pdf_path.stat().st_size
        }
    except Exception as e:
        print(f"Warning: Could not extract metadata from {pdf_path}: {e}")
        return {'filename': pdf_path.name}


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, any]:
    """Extract text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        page_count = len(doc)
        
        # Extract metadata while doc is open
        metadata = doc.metadata
        doc_metadata = {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', ''),
            'filename': pdf_path.name,
            'file_size': pdf_path.stat().st_size
        }
        
        # Extract text from all pages
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            full_text += text + "\n"
        
        doc.close()
        
        # Clean the extracted text
        cleaned_text = clean_text(full_text)
        
        return {
            'id': pdf_path.stem,
            'text': cleaned_text,
            'metadata': doc_metadata,
            'source_file': str(pdf_path),
            'page_count': page_count,
            'text_length': len(cleaned_text)
        }
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def process_directory(input_dir: Path, output_file: Path) -> None:
    """Process all PDF files in directory and save to JSONL."""
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            
            result = extract_text_from_pdf(pdf_path)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                processed_count += 1
                print(f"  ✓ Extracted {result['text_length']:,} characters")
            else:
                print(f"  ✗ Failed to process {pdf_path.name}")
    
    print(f"\nCompleted: {processed_count}/{len(pdf_files)} files processed")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from Employment Act PDFs")
    parser.add_argument('--in', dest='input_dir', required=True,
                        help='Input directory containing PDF files')
    parser.add_argument('--out', dest='output_file', required=True,
                        help='Output JSONL file path')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return
    
    process_directory(input_dir, output_file)


if __name__ == "__main__":
    main()