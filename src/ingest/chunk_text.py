#!/usr/bin/env python3
# python src/ingest/chunk_text.py --in data/processed/text.jsonl --out data/processed/chunks.jsonl
"""
Text chunking for processed Employment Act documents.
Chunks text into 800-1000 character segments with 150 character stride.
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - environment variables can be set directly
    pass

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


def detect_section_id(text: str) -> Optional[str]:
    """Detect section ID from text content with enhanced patterns."""
    # Enhanced patterns for Malaysian Employment Act sections
    patterns = [
        r'(?:Section|Sec\.?)\s*(\d+[A-Z]?(?:\([a-z]+\))?)',  # Section 60A(a)
        r'(\d+[A-Z]?(?:\([a-z]+\))?)\.\s*[A-Z]',  # 60A(a). Capital letter
        r'(\d+[A-Z]?)\s+[A-Z]',  # 60A Overtime  
        r'Part\s+([IVX]+)',  # Roman numeral parts
        r'Chapter\s+(\d+)',
        r'(\d+[A-Z]?)\s*[-–]\s*[A-Z]',  # 60A - Title format
        r'(\d+[A-Z]?)\s*[:\.]',  # 60A: or 60A.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"EA-{match.group(1)}"
    
    return None


def extract_section_title(text: str) -> Optional[str]:
    """Extract section title from text."""
    # Look for section titles after section numbers
    title_patterns = [
        r'(?:Section|Sec\.?)\s*\d+[A-Z]?(?:\([a-z]+\))?\s*[-–]?\s*([A-Z][^\n.]{10,80})',
        r'\d+[A-Z]?(?:\([a-z]+\))?\.\s*([A-Z][^\n.]{10,80})',
        r'\d+[A-Z]?(?:\([a-z]+\))?\s+([A-Z][^\n.]{10,80})',
        r'\d+[A-Z]?\s*[-–]\s*([A-Z][^\n.]{10,80})',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text[:200])  # Check first 200 chars
        if match:
            title = match.group(1).strip()
            # Clean up common suffixes
            title = re.sub(r'\s*[-–]\s*$', '', title)
            title = re.sub(r'\s*\.$', '', title)
            return title
    
    return None


def estimate_section_from_context(text: str, surrounding_chunks: List[str]) -> Optional[str]:
    """Estimate section ID from surrounding context if not found in current chunk."""
    # Look in surrounding chunks for section markers
    all_text = ' '.join(surrounding_chunks + [text])
    return detect_section_id(all_text)


def create_chunk_id(text: str, index: int) -> str:
    """Create a unique ID for the chunk."""
    # Use hash of content + index for uniqueness
    content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"chunk_{index:04d}_{content_hash}"


def chunk_text(text: str, chunk_size: int = 1000, stride: int = 300, min_chunk_size: int = 100, with_titles: bool = True) -> List[Dict[str, any]]:
    """
    Chunk text into overlapping segments.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size (800-1000 chars)
        stride: Overlap size between chunks
        min_chunk_size: Minimum chunk size to keep
    
    Returns:
        List of chunk dictionaries
    """
    # Fallback: if text is short but non-empty, emit a single chunk
    if len(text) < min_chunk_size:
        if len(text) == 0:
            return []
        chunk_text_value = text.strip()
        section_id = detect_section_id(chunk_text_value)
        return [{
            'chunk_id': create_chunk_id(chunk_text_value, 0),
            'text': chunk_text_value,
            'start_pos': 0,
            'end_pos': len(text),
            'length': len(chunk_text_value),
            'section_id': section_id,
            'chunk_index': 0
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this would be the last chunk and it's too small, extend the previous chunk
        if end >= len(text):
            end = len(text)
            if end - start < min_chunk_size and chunks:
                # Extend the previous chunk instead of creating a tiny one
                chunks[-1]['text'] += text[chunks[-1]['end_pos']:]
                chunks[-1]['end_pos'] = end
                chunks[-1]['length'] = len(chunks[-1]['text'])
                break
        else:
            # Try to end at sentence boundary
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:  # Only if we find a reasonable sentence break
                end = sentence_end + 1
            # Otherwise try paragraph boundary
            elif text.rfind('\n\n', start, end) > start:
                end = text.rfind('\n\n', start, end) + 2
            # Otherwise try line boundary
            elif text.rfind('\n', start, end) > start:
                end = text.rfind('\n', start, end) + 1
        
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) >= min_chunk_size:
            # Detect section ID and title for this chunk
            section_id = detect_section_id(chunk_text)
            section_title = extract_section_title(chunk_text)
            
            # Enhance chunk text with title context if available and requested
            enhanced_text = chunk_text
            if with_titles and section_title and section_id:
                # Prepend title if not already in chunk
                if section_title.lower() not in chunk_text[:100].lower():
                    enhanced_text = f"[Employment Act Section {section_id}: {section_title}] {chunk_text}"
            elif with_titles and section_id:
                # At minimum, add section ID
                enhanced_text = f"[Employment Act Section {section_id}] {chunk_text}"
            
            chunk = {
                'chunk_id': create_chunk_id(chunk_text, chunk_index),
                'text': enhanced_text,
                'original_text': chunk_text,
                'start_pos': start,
                'end_pos': end,
                'length': len(enhanced_text),
                'section_id': section_id,
                'section_title': section_title,
                'chunk_index': chunk_index
            }
            
            chunks.append(chunk)
            chunk_index += 1
        
        # Move start position forward by stride
        start += chunk_size - stride
        
        # If we've reached the end, break
        if end >= len(text):
            break
    
    return chunks


def post_process_chunks(chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """Post-process chunks to improve section ID detection."""
    for i, chunk in enumerate(chunks):
        if not chunk['section_id']:
            # Look for section ID in nearby chunks
            context_chunks = []
            
            # Get surrounding chunks for context
            start_idx = max(0, i - 2)
            end_idx = min(len(chunks), i + 3)
            
            for j in range(start_idx, end_idx):
                if j != i:
                    context_chunks.append(chunks[j]['text'])
            
            estimated_section = estimate_section_from_context(chunk['text'], context_chunks)
            if estimated_section:
                chunk['section_id'] = estimated_section
            
            # If still no section, try to inherit from previous chunk
            elif i > 0 and chunks[i-1]['section_id']:
                chunk['section_id'] = chunks[i-1]['section_id'] + '-cont'
    
    return chunks


def process_document(doc: Dict[str, any], chunk_size: int = 1000, stride: int = 300, with_titles: bool = True) -> List[Dict[str, any]]:
    """Process a single document and return its chunks."""
    text = doc['text']
    doc_id = doc['id']
    
    # Create chunks
    chunks = chunk_text(text, chunk_size, stride, with_titles=with_titles)
    
    # Post-process for better section detection
    chunks = post_process_chunks(chunks)
    
    # Add document-level metadata to each chunk
    for chunk in chunks:
        chunk.update({
            'document_id': doc_id,
            'source_file': doc.get('source_file', ''),
            'url': f"#{doc_id}#{chunk['section_id'] or 'unknown'}",
            'metadata': doc.get('metadata', {})
        })
    
    return chunks


def process_jsonl_file(input_file: Path, output_file: Path, chunk_size: int = 1000, stride: int = 300, with_titles: bool = True, force: bool = False) -> None:
    """Process JSONL file and create chunks with atomic writes."""
    print(f"Processing: {input_file}")
    print(f"Effective config - Chunk size: {chunk_size}, Stride: {stride}, With titles: {with_titles}")
    
    # Check if outputs are up-to-date
    metadata_file = output_file.with_suffix('.metadata.json')
    if check_up_to_date([output_file, metadata_file], [input_file], force):
        print(f"Output files are up-to-date. Use --force to regenerate.")
        return
    
    # Create metadata
    config = {
        'stage': 'chunk_text',
        'chunk_size': chunk_size,
        'stride': stride,
        'with_titles': with_titles,
        'min_chunk_size': 100
    }
    
    metadata = create_metadata(
        stage='chunk_text',
        input_files=[input_file],
        output_files=[output_file],
        config=config
    )
    
    all_chunks = []
    document_count = 0
    sections_with_id = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                chunks = process_document(doc, chunk_size, stride, with_titles)
                all_chunks.extend(chunks)
                document_count += 1
                
                # Count chunks with section IDs for coverage
                sections_with_id += sum(1 for chunk in chunks if chunk.get('section_id'))
                
                print(f"  Document {document_count}: {doc['id']} -> {len(chunks)} chunks")
                
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}")
    
    # Calculate section coverage
    section_coverage = (sections_with_id / len(all_chunks) * 100) if all_chunks else 0
    
    # Add chunking stats to metadata
    chunk_sizes = [chunk['length'] for chunk in all_chunks]
    metadata['chunking_stats'] = {
        'total_documents': document_count,
        'total_chunks': len(all_chunks),
        'avg_chunks_per_doc': len(all_chunks) / document_count if document_count > 0 else 0,
        'section_coverage_percent': section_coverage,
        'chunks_with_sections': sections_with_id,
        'chunk_size_stats': {
            'min': min(chunk_sizes) if chunk_sizes else 0,
            'max': max(chunk_sizes) if chunk_sizes else 0,
            'avg': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        }
    }
    
    # Write chunks atomically
    print(f"Writing {len(all_chunks)} chunks to {output_file}")
    atomic_write_jsonl(all_chunks, output_file)
    
    # Finalize and save metadata
    finalize_metadata(metadata, [output_file])
    save_metadata(metadata, metadata_file)
    
    print(f"\nSummary:")
    print(f"  Documents processed: {document_count}")
    print(f"  Total chunks created: {len(all_chunks)}")
    print(f"  Average chunks per document: {len(all_chunks)/document_count:.1f}")
    print(f"  Section coverage: {section_coverage:.1f}% ({sections_with_id}/{len(all_chunks)} chunks have section_id)")
    print(f"  Output saved to: {output_file}")
    print(f"  Metadata saved to: {metadata_file}")
    
    # Print chunk size distribution
    if chunk_sizes:
        print(f"  Chunk size stats:")
        print(f"    Min: {min(chunk_sizes)} chars")
        print(f"    Max: {max(chunk_sizes)} chars")
        print(f"    Average: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")


def main():
    parser = argparse.ArgumentParser(description="Chunk text from processed documents")
    parser.add_argument('--in', dest='input_file', required=True,
                        help='Input JSONL file from pdf_to_text.py')
    parser.add_argument('--out', dest='output_file', required=True,
                        help='Output JSONL file for chunks')
    parser.add_argument('--chunk', type=int, default=int(os.getenv('CHUNK_SIZE', '1000')),
                        help='Target chunk size in characters (default: from env or 1000)')
    parser.add_argument('--stride', type=int, default=int(os.getenv('CHUNK_STRIDE', '300')),
                        help='Overlap size between chunks (default: from env or 300)')
    # Titles toggle (both flags supported; defaults to True)
    parser.add_argument('--with-titles', dest='with_titles', action='store_true',
                        help='Include section titles in chunks (default)')
    parser.add_argument('--no-titles', dest='with_titles', action='store_false',
                        help='Do not include section titles in chunks')
    parser.set_defaults(with_titles=True)
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if outputs are up-to-date')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return
    
    process_jsonl_file(input_file, output_file, args.chunk, args.stride, getattr(args, 'with_titles', True), args.force)


if __name__ == "__main__":
    main()
