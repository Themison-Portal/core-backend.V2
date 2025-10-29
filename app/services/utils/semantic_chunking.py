"""
Semantic chunking that respects document structure (headers, numbered lists, etc.)
"""

import re
from typing import Any, Dict, List

from langchain_core.documents import Document


def chunk_text_semantic(
    content: str,
    metadata: Dict[str, Any] = None,
    chunk_size: int = 1500,  # Larger to fit complete sections
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunk content with semantic awareness:
    1. Never split numbered/bulleted lists
    2. Always include section headers in chunks
    3. Group related content together

    Args:
        content: Full document text
        metadata: Metadata to attach to chunks
        chunk_size: Target chunk size (will be exceeded to keep sections intact)
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document chunks with metadata
    """
    print(f"🔧 SEMANTIC CHUNKING CONFIG: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # Step 1: Identify sections (headers like "6.1 Inclusion Criteria")
    sections = _split_into_sections(content)
    print(f"📊 Found {len(sections)} sections")

    # Step 2: Process each section
    chunks = []
    for section in sections:
        section_chunks = _chunk_section(section, chunk_size, chunk_overlap, metadata)
        chunks.extend(section_chunks)

    print(f"📊 SEMANTIC CHUNKING RESULTS: {len(chunks)} chunks created from {len(content)} characters")
    if chunks:
        avg_size = sum(len(c.page_content) for c in chunks) / len(chunks)
        toc_count = sum(1 for c in chunks if c.metadata.get('is_toc', False))
        print(f"   - Average chunk size: {int(avg_size)} chars")
        print(f"   - Min chunk size: {min(len(c.page_content) for c in chunks)} chars")
        print(f"   - Max chunk size: {max(len(c.page_content) for c in chunks)} chars")
        print(f"   - TOC chunks: {toc_count} ({int(toc_count/len(chunks)*100)}%)")

    return chunks


def _split_into_sections(content: str) -> List[Dict[str, str]]:
    """
    Split content into sections based on headers.

    Detects patterns like:
    - "6 PATIENT SELECTION AND WITHDRAWAL"
    - "6.1 Inclusion Criteria All Cohorts"
    - "6.1.2 Subsubsection"
    - "Table 2 Flow Chart – Schedule of Assessments"
    - "Figure 1 Study Design"
    - "Appendix A Safety Guidelines"

    Returns list of dicts with 'header', 'content', 'parent_header', 'section_level'
    """
    # Pattern for headers (improved to support multi-level numbering and appendices):
    # 1. Numbered sections: "6.1.2 Inclusion Criteria" (supports any depth)
    # 2. Tables: "Table 2 Flow Chart – Schedule of Assessments"
    # 3. Figures: "Figure 1 Study Design"
    # 4. Appendices: "Appendix A Safety Guidelines"
    header_patterns = [
        (r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s\(\)\-–—:]+)', 'numbered'),  # Numbered sections (multi-level)
        (r'^(Table\s+\d+)\s*[-–—:]?\s*(.+)?', 'table'),  # Tables
        (r'^(Figure\s+\d+)\s*[-–—:]?\s*(.+)?', 'figure'),  # Figures
        (r'^(Appendix\s+[A-Z0-9]+)\s*[-–—:]?\s*(.+)?', 'appendix'),  # Appendices
    ]

    lines = content.split('\n')
    sections = []
    current_header = None
    current_content = []
    current_section_type = None
    current_section_start = 0
    char_position = 0  # Track absolute character position in document

    # Track parent headers (for hierarchical context)
    # Key: section level (e.g., "6" or "6.1"), Value: header text
    parent_headers = {}

    for line in lines:
        # Skip TOC entries: lines with pattern "text ........ number"
        # Pattern: 3+ consecutive dots followed by optional spaces and a number at end
        line_stripped = line.strip()
        if re.search(r'\.{3,}\s*\d+\s*$', line_stripped):
            current_content.append(line)
            char_position += len(line) + 1
            continue

        # Check if this line matches any header pattern
        is_header = False
        for pattern, section_type in header_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                is_header = True
                # Save previous section if exists
                if current_header is not None and sections:
                    # Update the last section's content
                    sections[-1]['content'] = '\n'.join(current_content).strip()

                # Extract section number and determine hierarchy
                section_number = match.group(1)
                section_title = match.group(2) if match.lastindex >= 2 else ''
                current_header = line.strip()
                current_section_type = section_type
                current_content = []
                current_section_start = char_position  # Save start position

                # Determine parent header for numbered sections
                parent_header = None
                section_level = 0

                if section_type == 'numbered':
                    # Extract hierarchy: "6.1.2" → levels = ["6", "6.1", "6.1.2"]
                    parts = section_number.split('.')
                    section_level = len(parts)

                    # Update parent_headers dict
                    for i in range(len(parts)):
                        level_key = '.'.join(parts[:i+1])
                        parent_headers[level_key] = '.'.join(parts[:i+1]) + ' ' + (section_title if i == len(parts)-1 else '')

                    # Find parent (one level up)
                    if section_level > 1:
                        parent_key = '.'.join(parts[:-1])
                        parent_header = parent_headers.get(parent_key)

                # Store section with metadata and start position
                sections.append({
                    'header': current_header,
                    'content': '',  # Will be filled as we process lines
                    'parent_header': parent_header,
                    'section_type': section_type,
                    'section_level': section_level,
                    'start_char': current_section_start  # Use saved start position
                })

                break

        if not is_header:
            # Add to current section content
            current_content.append(line)

        # Update character position (line + newline)
        char_position += len(line) + 1

    # Update last section's content
    if sections and current_content:
        sections[-1]['content'] = '\n'.join(current_content).strip()

    # If no sections found, treat entire content as one section
    if not sections:
        sections = [{
            'header': 'Document',
            'content': content,
            'parent_header': None,
            'section_type': 'document',
            'section_level': 0,
            'start_char': 0
        }]

    return sections


def _is_toc_section(header: str, content: str) -> bool:
    """
    Detect if a section is Table of Contents / List of Contents.

    TOC characteristics:
    1. Header contains "Table of Contents", "List of Contents", "Contents", "TABLE OF CONTENTS", etc.
    2. Content has multiple lines with pattern: "text ......... page_number"
    3. High density of dots (.) in the content
    """
    # Check header for TOC keywords
    header_lower = header.lower()
    toc_keywords = ['table of contents', 'list of contents', 'contents', 'table of content']

    if any(keyword in header_lower for keyword in toc_keywords):
        return True

    # Check content for TOC pattern
    # Count lines with pattern: text + many dots + number
    lines = content.split('\n')
    toc_pattern_count = 0

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Pattern: has multiple dots (5+) and ends with a number
        if line_stripped.count('.') >= 5:
            # Check if line ends with a number (page number)
            if re.search(r'\d+\s*$', line_stripped):
                toc_pattern_count += 1

    # If more than 30% of non-empty lines match TOC pattern, it's a TOC
    non_empty_lines = [l for l in lines if l.strip()]
    if non_empty_lines and toc_pattern_count / len(non_empty_lines) > 0.3:
        return True

    return False


def _chunk_section(
    section: Dict[str, str],
    chunk_size: int,
    chunk_overlap: int,
    metadata: Dict[str, Any]
) -> List[Document]:
    """
    Chunk a single section while:
    1. Always including the header (and parent header if exists)
    2. Never splitting numbered lists
    3. Respecting chunk_size as a soft limit
    """
    header = section['header']
    content = section['content']
    parent_header = section.get('parent_header')
    section_type = section.get('section_type', 'unknown')
    section_level = section.get('section_level', 0)
    section_start_char = section.get('start_char', 0)  # Absolute position in document

    # Build full context header (parent + current)
    full_header = header
    if parent_header:
        full_header = f"{parent_header}\n{header}"

    # Detect if this is a TOC section
    is_toc = _is_toc_section(header, content)

    if is_toc:
        print(f"📋 TOC DETECTED: Section '{header[:50]}...' marked as TOC")
        print(f"   Content preview: {content[:100]}...")

    # Add to metadata
    chunk_metadata = {
        **(metadata or {}),
        'section_header': header,
        'parent_section': parent_header,
        'section_type': section_type,
        'section_level': section_level,
        'is_toc': is_toc
    }

    # Detect if this section contains a numbered list
    has_numbered_list = _contains_numbered_list(content)

    if has_numbered_list:
        # Split into list items
        items = _split_numbered_list(content)

        # Group items into chunks, always including full header
        return _group_list_items_into_chunks(full_header, items, chunk_size, chunk_overlap, chunk_metadata, section_start_char)
    else:
        # Regular text - chunk with full header prepended
        return _chunk_regular_text(full_header, content, chunk_size, chunk_overlap, chunk_metadata, section_start_char)


def _contains_numbered_list(content: str) -> bool:
    """Check if content contains numbered list (1. 2. 3. or 1) 2) 3))"""
    # Pattern: start of line, optional whitespace, number, dot or paren, space
    pattern = r'^\s*\d+[\.\)]\s+'
    return bool(re.search(pattern, content, re.MULTILINE))


def _split_numbered_list(content: str) -> List[str]:
    """
    Split content by numbered list items.

    Example input:
    "1. First item
    Some text
    2. Second item"

    Returns: ["1. First item\nSome text", "2. Second item"]
    """
    # Pattern: start of line, optional whitespace, number, dot or paren
    pattern = r'^\s*(\d+[\.\)])\s+'

    items = []
    current_item = []

    for line in content.split('\n'):
        if re.match(pattern, line):
            # Save previous item
            if current_item:
                items.append('\n'.join(current_item))
            # Start new item
            current_item = [line]
        else:
            # Continue current item
            current_item.append(line)

    # Add last item
    if current_item:
        items.append('\n'.join(current_item))

    return items


def _group_list_items_into_chunks(
    header: str,
    items: List[str],
    chunk_size: int,
    chunk_overlap: int,
    metadata: Dict[str, Any],
    section_start_char: int = 0
) -> List[Document]:
    """
    Group list items into chunks, always including header.

    Strategy:
    - Always prepend header to each chunk
    - Group items until chunk_size is reached
    - NEVER split an item across chunks
    - Add overlap by including last N items from previous chunk
    """
    chunks = []
    current_chunk_items = []
    current_size = len(header) + 2  # +2 for newlines
    header_offset = len(header) + 2
    current_position = 0  # Position within section content

    for item in items:
        item_size = len(item) + 1  # +1 for newline

        # If adding this item exceeds chunk_size AND we have items, create chunk
        if current_chunk_items and (current_size + item_size > chunk_size):
            # Create chunk with header + items
            chunk_text = header + '\n\n' + '\n'.join(current_chunk_items)

            # Calculate absolute start position for this chunk
            absolute_start_index = section_start_char + header_offset + current_position

            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **metadata,
                    'chunk_type': 'numbered_list',
                    'start_index': absolute_start_index
                }
            ))

            # Update position tracker
            current_position += sum(len(i) + 1 for i in current_chunk_items)

            # Start new chunk with overlap (last 1-2 items)
            overlap_items = _get_overlap_items(current_chunk_items, chunk_overlap)
            current_chunk_items = overlap_items + [item]
            current_size = len(header) + 2 + sum(len(i) + 1 for i in current_chunk_items)
        else:
            # Add item to current chunk
            current_chunk_items.append(item)
            current_size += item_size

    # Add last chunk
    if current_chunk_items:
        chunk_text = header + '\n\n' + '\n'.join(current_chunk_items)

        # Calculate absolute start position for last chunk
        absolute_start_index = section_start_char + header_offset + current_position

        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                **metadata,
                'chunk_type': 'numbered_list',
                'start_index': absolute_start_index
            }
        ))

    return chunks


def _get_overlap_items(items: List[str], overlap_size: int) -> List[str]:
    """Get last N items from list to use as overlap in next chunk"""
    if not items:
        return []

    # Calculate how many items fit in overlap_size
    overlap_items = []
    total_size = 0

    for item in reversed(items):
        item_size = len(item) + 1
        if total_size + item_size <= overlap_size:
            overlap_items.insert(0, item)
            total_size += item_size
        else:
            break

    return overlap_items


def _chunk_regular_text(
    header: str,
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Dict[str, Any],
    section_start_char: int = 0
) -> List[Document]:
    """
    Chunk regular text (non-list) while always including header in EVERY chunk.
    Uses paragraph-aware splitting.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # First, chunk the content WITHOUT header to get logical breaks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size - len(header) - 2,  # Reserve space for header
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        add_start_index=True
    )

    doc = Document(
        page_content=content,
        metadata=metadata or {}
    )

    # Split content into chunks
    content_chunks = splitter.split_documents([doc])

    # Now prepend header to EACH chunk
    result_chunks = []
    header_offset = len(header) + 2  # header + "\n\n"

    for chunk in content_chunks:
        # Get start_index from splitter (relative to content)
        relative_start_index = chunk.metadata.get('start_index', 0)

        # Calculate absolute start_index in document
        # section_start_char + len(header line) + relative position in content
        absolute_start_index = section_start_char + header_offset + relative_start_index

        chunk_with_header = Document(
            page_content=f"{header}\n\n{chunk.page_content}",
            metadata={
                **metadata,
                'chunk_type': 'regular_text',
                'start_index': absolute_start_index  # Store absolute position
            }
        )
        result_chunks.append(chunk_with_header)

    return result_chunks
