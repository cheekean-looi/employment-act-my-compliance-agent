#!/usr/bin/env python3
# python src/ingest/pdf_to_text.py --in data/raw_pdfs --out data/processed/text.jsonl
"""
PDF to text extraction with header/footer stripping.
Processes Employment Act Malaysia PDFs and amendments.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

try:
    from .utils import (
        create_metadata, finalize_metadata, save_metadata, 
        atomic_write_jsonl, check_up_to_date
    )
except ImportError:
    # When run as script, use absolute import
    from utils import (
        create_metadata, finalize_metadata, save_metadata, 
        atomic_write_jsonl, check_up_to_date
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Install with: pip install PyMuPDF")
    exit(1)


def detect_repeated_blocks(doc) -> Set[str]:
    """Detect text blocks that repeat across multiple pages (headers/footers)."""
    block_counts = defaultdict(int)
    page_count = len(doc)
    
    # If only one page, can't detect repeats
    if page_count <= 1:
        return set()
    
    # Collect normalized text blocks from all pages
    for page_num in range(page_count):
        try:
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            
            for block in blocks:
                if len(block) >= 4:  # Ensure block has text content
                    text = block[4].strip().lower()
                    # Normalize text for comparison
                    normalized = re.sub(r'\s+', ' ', text)
                    if len(normalized) > 5 and len(normalized) < 100:  # Reasonable header/footer size
                        block_counts[normalized] += 1
        except Exception as e:
            logging.warning(f"Error processing page {page_num} for repeat detection: {e}")
            continue
    
    # Find blocks that appear on multiple pages (likely headers/footers)
    repeated_threshold = max(2, page_count // 3)  # Appear on at least 1/3 of pages
    repeated_blocks = {text for text, count in block_counts.items() if count >= repeated_threshold}
    
    return repeated_blocks


def extract_text_with_positional_filtering(doc, repeated_blocks: Set[str], top_band_pct: float = 0.08, bottom_band_pct: float = 0.08) -> str:
    """Extract text using positional filtering to remove headers/footers.
    
    Args:
        doc: PyMuPDF document
        repeated_blocks: Set of repeated text blocks to filter
        top_band_pct: Percentage of page height to treat as header band (default 8%)
        bottom_band_pct: Percentage of page height to treat as footer band (default 8%)
    """
    all_text = []
    page_count = len(doc)
    
    for page_num in range(page_count):
        try:
            page = doc.load_page(page_num)
            page_rect = page.rect
            page_height = page_rect.height
            
            # Define header/footer bands using configurable percentages
            top_band = page_height * top_band_pct
            bottom_band = page_height * (1.0 - bottom_band_pct)
            
            # Get text blocks with position information
            blocks = page.get_text("blocks")
            
            page_text_blocks = []
            for block in blocks:
                if len(block) >= 5:  # Block format: (x0, y0, x1, y1, text, block_no, block_type)
                    x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                    
                    # Skip blocks in header/footer bands
                    if y0 < top_band or y1 > bottom_band:
                        continue
                    
                    # Skip repeated blocks (headers/footers)
                    normalized_text = re.sub(r'\s+', ' ', text.strip().lower())
                    if normalized_text in repeated_blocks:
                        continue
                    
                    # Clean and add block text
                    clean_block_text = text.strip()
                    if clean_block_text and len(clean_block_text) > 5:
                        page_text_blocks.append(clean_block_text)
            
            # Join blocks for this page
            page_text = '\n'.join(page_text_blocks)
            if page_text.strip():
                all_text.append(page_text)
                
        except Exception as e:
            logging.error(f"Error processing page {page_num}: {e}")
            # Fallback to simple text extraction for this page
            try:
                page = doc.load_page(page_num)
                fallback_text = page.get_text()
                if fallback_text.strip():
                    all_text.append(clean_text_simple(fallback_text))
            except Exception as e2:
                logging.error(f"Fallback extraction also failed for page {page_num}: {e2}")
                continue
    
    return '\n\n'.join(all_text)


def clean_text_simple(text: str) -> str:
    """Simple text cleaning as fallback."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip page numbers and very short administrative text
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


def extract_text_from_pdf(pdf_path: Path, top_band_pct: float = 0.08, bottom_band_pct: float = 0.08) -> Dict[str, any]:
    """Extract text from a single PDF file using positional filtering.
    
    Args:
        pdf_path: Path to PDF file
        top_band_pct: Percentage of page height to treat as header band (default 8%)
        bottom_band_pct: Percentage of page height to treat as footer band (default 8%)
    """
    try:
        doc = fitz.open(pdf_path)
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
        
        # Detect repeated blocks (headers/footers) across pages
        print(f"  Analyzing page structure for header/footer detection...")
        repeated_blocks = detect_repeated_blocks(doc)
        if repeated_blocks:
            print(f"  Found {len(repeated_blocks)} repeated header/footer patterns")
        
        # Extract text using positional filtering
        print(f"  Extracting text with positional filtering (top: {top_band_pct*100}%, bottom: {bottom_band_pct*100}%)...")
        cleaned_text = extract_text_with_positional_filtering(doc, repeated_blocks, top_band_pct, bottom_band_pct)
        
        doc.close()
        
        return {
            'id': pdf_path.stem,
            'text': cleaned_text,
            'metadata': doc_metadata,
            'source_file': str(pdf_path),
            'page_count': page_count,
            'text_length': len(cleaned_text),
            'extraction_method': 'positional_filtering',
            'repeated_patterns_detected': len(repeated_blocks)
        }
        
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        print(f"Error processing {pdf_path}: {e}")
        return None


def process_directory(input_dir: Path, output_file: Path, force: bool = False, top_band_pct: float = 0.08, bottom_band_pct: float = 0.08, limit: Optional[int] = None) -> None:
    """Process all PDF files in directory and save to JSONL with atomic writes."""
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    original_count = len(pdf_files)
    if limit is not None and limit > 0:
        pdf_files = pdf_files[:min(limit, original_count)]
        print(f"Found {original_count} PDF files; limiting to first {len(pdf_files)} for dev mode")
    else:
        print(f"Found {original_count} PDF files to process")
    
    # Check if outputs are up-to-date
    metadata_file = output_file.with_suffix('.metadata.json')
    if check_up_to_date([output_file, metadata_file], pdf_files, force):
        print(f"Output files are up-to-date. Use --force to regenerate.")
        return
    
    # Create metadata
    config = {
        'stage': 'pdf_to_text',
        'input_directory': str(input_dir),
        'extraction_method': 'positional_filtering',
        'header_footer_bands': {'top_pct': top_band_pct, 'bottom_pct': bottom_band_pct},
        'limit': limit
    }
    
    metadata = create_metadata(
        stage='pdf_to_text',
        input_files=pdf_files,
        output_files=[output_file],
        config=config
    )
    
    processed_count = 0
    all_results = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        result = extract_text_from_pdf(pdf_path, top_band_pct, bottom_band_pct)
        if result:
            all_results.append(result)
            processed_count += 1
            print(f"  ✓ Extracted {result['text_length']:,} characters")
        else:
            print(f"  ✗ Failed to process {pdf_path.name}")
    
    # Add processing stats to metadata
    metadata['processing_stats'] = {
        'total_files': len(pdf_files),
        'processed_files': processed_count,
        'failed_files': len(pdf_files) - processed_count,
        'total_characters': sum(r['text_length'] for r in all_results),
        'total_pages': sum(r['page_count'] for r in all_results)
    }
    
    # Write results atomically
    print(f"Writing {len(all_results)} results to {output_file}")
    atomic_write_jsonl(all_results, output_file)
    
    # Finalize and save metadata
    finalize_metadata(metadata, [output_file])
    save_metadata(metadata, metadata_file)
    
    print(f"\nCompleted: {processed_count}/{len(pdf_files)} files processed")
    print(f"Output saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from Employment Act PDFs")
    parser.add_argument('--in', dest='input_dir', required=True,
                        help='Input directory containing PDF files')
    parser.add_argument('--out', dest='output_file', required=True,
                        help='Output JSONL file path')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if outputs are up-to-date')
    parser.add_argument('--top-band', type=float, default=0.08,
                        help='Top header band percentage (default: 0.08 = 8%%)')
    parser.add_argument('--bottom-band', type=float, default=0.08,
                        help='Bottom footer band percentage (default: 0.08 = 8%%)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of PDFs to process (dev/testing)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return
    
    process_directory(
        input_dir,
        output_file,
        args.force,
        getattr(args, 'top_band', 0.08),
        getattr(args, 'bottom_band', 0.08),
        getattr(args, 'limit', None)
    )


if __name__ == "__main__":
    main()
