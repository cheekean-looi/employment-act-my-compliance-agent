#!/usr/bin/env python3
# python src/ingest/chunk_text.py --in data/processed/text.jsonl --out data/processed/chunks.jsonl
"""
Text chunking for processed Employment Act documents.
Chunks text into 800-1000 character segments with 150 character stride.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


def detect_section_id(text: str) -> Optional[str]:
    """Detect section ID from text content."""
    # Common patterns for Malaysian Employment Act sections
    patterns = [
        r'(?:Section|Sec\.?)\s*(\d+[A-Z]?(?:\(\d+\))?)',
        r'(\d+[A-Z]?(?:\(\d+\))?)\.\s*[A-Z]',  # Section number followed by period and capital letter
        r'Part\s+([IVX]+)',  # Roman numeral parts
        r'Chapter\s+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"EA-{match.group(1)}"
    
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


def chunk_text(text: str, chunk_size: int = 900, stride: int = 150, min_chunk_size: int = 100) -> List[Dict[str, any]]:
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
            # Detect section ID for this chunk
            section_id = detect_section_id(chunk_text)
            
            chunk = {
                'chunk_id': create_chunk_id(chunk_text, chunk_index),
                'text': chunk_text,
                'start_pos': start,
                'end_pos': end,
                'length': len(chunk_text),
                'section_id': section_id,
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


def process_document(doc: Dict[str, any], chunk_size: int = 900, stride: int = 150) -> List[Dict[str, any]]:
    """Process a single document and return its chunks."""
    text = doc['text']
    doc_id = doc['id']
    
    # Create chunks
    chunks = chunk_text(text, chunk_size, stride)
    
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


def process_jsonl_file(input_file: Path, output_file: Path, chunk_size: int = 900, stride: int = 150) -> None:
    """Process JSONL file and create chunks."""
    print(f"Processing: {input_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    document_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                chunks = process_document(doc, chunk_size, stride)
                all_chunks.extend(chunks)
                document_count += 1
                
                print(f"  Document {document_count}: {doc['id']} -> {len(chunks)} chunks")
                
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}")
    
    # Write all chunks to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"\nSummary:")
    print(f"  Documents processed: {document_count}")
    print(f"  Total chunks created: {len(all_chunks)}")
    print(f"  Average chunks per document: {len(all_chunks)/document_count:.1f}")
    print(f"  Output saved to: {output_file}")
    
    # Print chunk size distribution
    chunk_sizes = [chunk['length'] for chunk in all_chunks]
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
    parser.add_argument('--chunk', type=int, default=900,
                        help='Target chunk size in characters (default: 900)')
    parser.add_argument('--stride', type=int, default=150,
                        help='Overlap size between chunks (default: 150)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return
    
    process_jsonl_file(input_file, output_file, args.chunk, args.stride)


if __name__ == "__main__":
    main()
