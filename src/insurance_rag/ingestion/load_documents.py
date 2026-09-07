from pathlib import Path
from collections import Counter
import json
import re

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling.datamodel.base_models import InputFormat

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    HeadingHierarchyOptions,
)

from docling_core.transforms.chunker.hybrid_chunker import (
    HybridChunker,
)

from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

from transformers import AutoTokenizer

from langchain_core.documents import Document


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

MAX_TOKENS = 400


# =============================================================================
# STEP 1 — PARSE PDF
# =============================================================================

def parse_pdf(pdf_path: str):
    """
    Parse PDF using Docling with heading hierarchy enabled.
    """

    heading_options = HeadingHierarchyOptions(
        enabled=True,
        use_bookmarks=True,
        use_numbering=True,
        use_style=True,
        use_font_style=True,
        max_level=6,
    )

    pipeline_options = PdfPipelineOptions(
        heading_hierarchy_options=heading_options,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(
        pdf_path
    )

    return result.document


# =============================================================================
# STEP 2 — CREATE TOKENIZER / HYBRID CHUNKER
# =============================================================================

def build_chunker(
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    max_tokens: int = MAX_TOKENS,
):
    """
    Create Docling HybridChunker using the tokenizer belonging
    to the embedding model.
    """

    hf_tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_id
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=max_tokens,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=False,
    )

    return chunker


# =============================================================================
# STEP 3 — GET TEXT FROM DOCLING ITEM
# =============================================================================

def get_item_text(item) -> str:
    """
    Safely extract text from a Docling item.
    """

    text = getattr(
        item,
        "text",
        "",
    )

    if text:
        return str(text).strip()

    return ""


# =============================================================================
# STEP 4 — GET PAGE NUMBERS
# =============================================================================

def get_page_numbers_from_items(
    doc_items,
) -> list[int]:
    """
    Return all PDF page numbers associated with the supplied
    Docling items.
    """

    page_numbers = set()

    for item in doc_items:

        provenance = getattr(
            item,
            "prov",
            [],
        ) or []

        for prov in provenance:

            page_no = getattr(
                prov,
                "page_no",
                None,
            )

            if page_no is not None:
                page_numbers.add(
                    page_no
                )

    return sorted(
        page_numbers
    )


# =============================================================================
# STEP 5 — NORMALIZE HEADING TEXT
# =============================================================================

def normalize_heading(text: str) -> str:
    """
    Normalize heading text so that repeated page headers can be detected.

    Example:

        " my: Optima Secure "
        "my:  Optima Secure"

    become equivalent.
    """

    text = str(
        text
    ).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.lower()


# =============================================================================
# STEP 6 — IDENTIFY HEADING LEVEL
# =============================================================================

def get_heading_level(item):
    """
    Return heading level for a Docling section header.

    Returns None for non-heading items.
    """

    label = str(
        getattr(
            item,
            "label",
            "",
        )
    ).lower()

    if label != "section_header":
        return None

    level = getattr(
        item,
        "level",
        None,
    )

    if level is None:
        return 1

    return int(
        level
    )


# =============================================================================
# STEP 7 — FIND REPEATED PAGE HEADERS
# =============================================================================

def find_repeated_headers(
    document,
) -> set[str]:
    """
    Find headings that are likely page headers rather than
    real document sections.

    We look for section-header items that appear on multiple
    different PDF pages.

    Example:

        my: Optima Secure

    appears on pages:

        2
        3
        4
        5
        6
        ...

    Therefore it is almost certainly a page header and should
    NOT reset the section hierarchy.
    """

    header_pages = {}

    for item, _level in document.iterate_items():

        heading_level = get_heading_level(
            item
        )

        if heading_level is None:
            continue

        text = get_item_text(
            item
        )

        if not text:
            continue

        normalized = normalize_heading(
            text
        )

        pages = get_page_numbers_from_items(
            [item]
        )

        if not pages:
            continue

        if normalized not in header_pages:

            header_pages[
                normalized
            ] = {
                "text": text,
                "pages": set(),
            }

        header_pages[
            normalized
        ]["pages"].update(
            pages
        )

    repeated_headers = set()

    for normalized, info in (
        header_pages.items()
    ):

        # Appears on at least 2 different pages.
        if len(info["pages"]) >= 2:

            repeated_headers.add(
                normalized
            )

    return repeated_headers


# =============================================================================
# STEP 8 — BUILD PARENT / CHILD SECTIONS
# =============================================================================

def build_sections(
    document,
):
    """
    Build explicit hierarchical sections.

    Important:

    Repeated page headers are ignored.

    Example:

        SECTION A. DEFINITIONS
            1.1. Standard Definitions
                Def. 1
                Def. 2
                Def. 3

    If "my: Optima Secure" appears on every page, it will NOT
    become a new heading.

    The resulting section therefore remains:

        SECTION A. DEFINITIONS
        1.1. Standard Definitions
    """

    sections = []

    # -------------------------------------------------------------------------
    # Identify repeated page headers FIRST.
    # -------------------------------------------------------------------------

    repeated_headers = find_repeated_headers(
        document
    )

    if repeated_headers:

        print(
            "\nIgnoring repeated page headers:"
        )

        for header in sorted(
            repeated_headers
        ):

            print(
                f"  - {header}"
            )

    # -------------------------------------------------------------------------
    # Current heading hierarchy.
    #
    # Example:
    #
    # {
    #     1: "SECTION A. DEFINITIONS",
    #     2: "1.1. Standard Definitions"
    # }
    # -------------------------------------------------------------------------

    heading_stack = {}

    current_items = []

    def flush_section():

        nonlocal current_items

        if not current_items:
            return

        headings = [
            heading_stack[level]
            for level in sorted(
                heading_stack
            )
            if heading_stack[level]
        ]

        sections.append(
            {
                "headings": headings.copy(),
                "items": current_items.copy(),
            }
        )

        current_items = []

    # -------------------------------------------------------------------------
    # Walk through Docling document.
    # -------------------------------------------------------------------------

    for item, _level in document.iterate_items():

        heading_level = get_heading_level(
            item
        )

        # =====================================================================
        # HEADING
        # =====================================================================

        if heading_level is not None:

            heading_text = get_item_text(
                item
            )

            if not heading_text:
                continue

            normalized_heading = (
                normalize_heading(
                    heading_text
                )
            )

            # -------------------------------------------------------------
            # IMPORTANT:
            #
            # Ignore repeated page headers.
            #
            # This prevents:
            #
            #     my: Optima Secure
            #
            # from resetting:
            #
            #     SECTION A. DEFINITIONS
            #         1.1. Standard Definitions
            # -------------------------------------------------------------

            if normalized_heading in (
                repeated_headers
            ):
                continue

            # -------------------------------------------------------------
            # Real heading.
            #
            # Content before this heading belongs to the previous section.
            # -------------------------------------------------------------

            flush_section()

            # -------------------------------------------------------------
            # Remove old headings at this level and below.
            # -------------------------------------------------------------

            levels_to_remove = [
                level
                for level in heading_stack
                if level >= heading_level
            ]

            for level in levels_to_remove:

                del heading_stack[
                    level
                ]

            # -------------------------------------------------------------
            # Add new heading.
            # -------------------------------------------------------------

            heading_stack[
                heading_level
            ] = heading_text

            continue

        # =====================================================================
        # NORMAL DOCUMENT ITEM
        # =====================================================================

        current_items.append(
            item
        )

    # Flush final section.
    flush_section()

    return sections


# =============================================================================
# STEP 9 — DETECT LOGICAL UNITS
# =============================================================================

def is_definition(
    text: str,
) -> bool:
    """
    Detect insurance definitions.

    Examples:

        Def. 1. Accident means...
        Def. 2. Any one illness means...
        Definition 1...
    """

    return bool(
        re.match(
            r"^\s*(Def\.|Definition)\s*\d+",
            text,
            flags=re.IGNORECASE,
        )
    )


def is_numbered_clause(
    text: str,
) -> bool:
    """
    Detect numbered clauses.

    Examples:

        1. The Company shall...
        2. The Policyholder shall...
        1.1. The Company...
        1.1.1. The Company...
    """

    return bool(
        re.match(
            r"^\s*\d+(?:\.\d+)*\.\s+",
            text,
        )
    )


def is_lettered_clause(
    text: str,
) -> bool:
    """
    Detect lettered clauses.

    Examples:

        (a) The Company shall...
        (b) The Policyholder shall...
        a) The Company shall...
    """

    return bool(
        re.match(
            r"^\s*(?:\([a-zA-Z]\)|[a-zA-Z]\))\s+",
            text,
        )
    )


# =============================================================================
# STEP 10 — SPLIT INTO LOGICAL UNITS
# =============================================================================

def split_into_logical_units(
    items,
) -> list[dict]:
    """
    Convert Docling items into logical units.

    Definitions, clauses and tables are treated as independent
    units and are not combined with unrelated text.

    Continuation paragraphs after a definition remain part of
    that definition until the next logical unit begins.
    """

    units = []

    current_text_parts = []
    current_items = []

    def flush_text():

        nonlocal current_text_parts
        nonlocal current_items

        if not current_text_parts:
            return

        text = "\n\n".join(
            current_text_parts
        ).strip()

        if text:

            units.append(
                {
                    "text": text,
                    "items": current_items.copy(),
                    "is_table": False,
                }
            )

        current_text_parts = []
        current_items = []

    for item in items:

        label = str(
            getattr(
                item,
                "label",
                "",
            )
        ).lower()

        # =====================================================================
        # TABLE
        # =====================================================================

        if label == "table":

            flush_text()

            try:

                table_text = (
                    item.export_to_markdown()
                )

            except Exception:

                table_text = ""

            if table_text:

                units.append(
                    {
                        "text": table_text.strip(),
                        "items": [item],
                        "is_table": True,
                    }
                )

            continue

        # =====================================================================
        # NORMAL TEXT
        # =====================================================================

        text = get_item_text(
            item
        )

        if not text:
            continue

        # =====================================================================
        # NEW DEFINITION
        # =====================================================================

        if is_definition(text):

            flush_text()

            units.append(
                {
                    "text": text,
                    "items": [item],
                    "is_table": False,
                }
            )

            continue

        # =====================================================================
        # NEW NUMBERED CLAUSE
        # =====================================================================

        if is_numbered_clause(text):

            flush_text()

            units.append(
                {
                    "text": text,
                    "items": [item],
                    "is_table": False,
                }
            )

            continue

        # =====================================================================
        # NEW LETTERED CLAUSE
        # =====================================================================

        if is_lettered_clause(text):

            flush_text()

            units.append(
                {
                    "text": text,
                    "items": [item],
                    "is_table": False,
                }
            )

            continue

        # =====================================================================
        # CONTINUATION / NORMAL PARAGRAPH
        # =====================================================================

        current_text_parts.append(
            text
        )

        current_items.append(
            item
        )

    # Flush final content.
    flush_text()

    return units


# =============================================================================
# STEP 11 — SPLIT OVERSIZED LOGICAL UNIT
# =============================================================================

def split_oversized_unit(
    text: str,
    chunker,
    max_tokens: int,
) -> list[str]:
    """
    Split an individual logical unit only when it exceeds max_tokens.

    This is a last-resort operation.

    Example:

        Def. 1 = 150 tokens
            → remains intact

        Def. 2 = 650 tokens
            → split because it cannot fit in one chunk
    """

    token_ids = (
        chunker.tokenizer.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )

    if len(token_ids) <= max_tokens:

        return [
            text
        ]

    pieces = []

    start = 0

    while start < len(token_ids):

        end = min(
            start + max_tokens,
            len(token_ids),
        )

        piece_ids = token_ids[
            start:end
        ]

        piece = (
            chunker
            .tokenizer
            .tokenizer
            .decode(
                piece_ids,
                skip_special_tokens=True,
            )
        ).strip()

        if piece:

            pieces.append(
                piece
            )

        start = end

    return pieces


# =============================================================================
# STEP 12 — CHUNK ONE SECTION
# =============================================================================

def chunk_section(
    section,
    chunker,
    max_tokens: int,
):
    """
    Chunk one hierarchical section.

    Important:

    The section's complete heading hierarchy is stored on EVERY
    resulting chunk.

    Example:

        SECTION A. DEFINITIONS
        1.1. Standard Definitions

    remains attached even when the section crosses pages.
    """

    headings = section[
        "headings"
    ]

    items = section[
        "items"
    ]

    if not items:

        return []

    section_path = (
        " > ".join(
            headings
        )
    )

    logical_units = (
        split_into_logical_units(
            items
        )
    )

    if not logical_units:

        return []

    chunks = []

    current_units = []
    current_items = []
    current_tokens = 0
    current_is_table = False

    # -------------------------------------------------------------------------
    # Process logical units.
    # -------------------------------------------------------------------------

    for unit in logical_units:

        unit_text = unit[
            "text"
        ]

        unit_items = unit[
            "items"
        ]

        unit_is_table = unit[
            "is_table"
        ]

        unit_tokens = (
            chunker.tokenizer.count_tokens(
                text=unit_text
            )
        )

        # =====================================================================
        # CASE 1 — LOGICAL UNIT ITSELF IS TOO LARGE
        # =====================================================================

        if unit_tokens > max_tokens:

            # Flush current chunk first.
            if current_units:

                chunks.append(
                    {
                        "text": "\n\n".join(
                            current_units
                        ),
                        "headings": headings.copy(),
                        "section_path": section_path,
                        "items": current_items.copy(),
                        "is_table": current_is_table,
                    }
                )

            current_units = []
            current_items = []
            current_tokens = 0
            current_is_table = False

            # -------------------------------------------------------------
            # Split oversized logical unit.
            # -------------------------------------------------------------

            pieces = (
                split_oversized_unit(
                    text=unit_text,
                    chunker=chunker,
                    max_tokens=max_tokens,
                )
            )

            for piece in pieces:

                chunks.append(
                    {
                        "text": piece,
                        "headings": headings.copy(),
                        "section_path": section_path,
                        "items": unit_items.copy(),
                        "is_table": unit_is_table,
                    }
                )

            continue

        # =====================================================================
        # CASE 2 — UNIT FITS INTO CURRENT CHUNK
        # =====================================================================

        if (
            current_tokens
            + unit_tokens
            <= max_tokens
        ):

            current_units.append(
                unit_text
            )

            current_items.extend(
                unit_items
            )

            current_tokens += (
                unit_tokens
            )

            current_is_table = (
                current_is_table
                or unit_is_table
            )

            continue

        # =====================================================================
        # CASE 3 — UNIT DOES NOT FIT
        # =====================================================================

        if current_units:

            chunks.append(
                {
                    "text": "\n\n".join(
                        current_units
                    ),
                    "headings": headings.copy(),
                    "section_path": section_path,
                    "items": current_items.copy(),
                    "is_table": current_is_table,
                }
            )

        # Start new chunk with this logical unit.
        current_units = [
            unit_text
        ]

        current_items = (
            unit_items.copy()
        )

        current_tokens = (
            unit_tokens
        )

        current_is_table = (
            unit_is_table
        )

    # -------------------------------------------------------------------------
    # Flush final chunk.
    # -------------------------------------------------------------------------

    if current_units:

        chunks.append(
            {
                "text": "\n\n".join(
                    current_units
                ),
                "headings": headings.copy(),
                "section_path": section_path,
                "items": current_items.copy(),
                "is_table": current_is_table,
            }
        )

    return chunks


# =============================================================================
# STEP 13 — BUILD LANGCHAIN DOCUMENTS
# =============================================================================

def build_documents(
    pdf_path: str,
    policy_name: str | None = None,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    max_tokens: int = MAX_TOKENS,
) -> list[Document]:
    """
    Convert one PDF into LangChain Documents.
    """

    pdf_path = Path(
        pdf_path
    )

    policy_name = (
        policy_name
        or pdf_path.stem
    )

    # -------------------------------------------------------------------------
    # Parse PDF
    # -------------------------------------------------------------------------

    print(
        f"\nParsing: {pdf_path.name}"
    )

    document = parse_pdf(
        str(pdf_path)
    )

    # -------------------------------------------------------------------------
    # Build tokenizer
    # -------------------------------------------------------------------------

    chunker = build_chunker(
        embedding_model_id=embedding_model_id,
        max_tokens=max_tokens,
    )

    # -------------------------------------------------------------------------
    # Build sections
    # -------------------------------------------------------------------------

    sections = build_sections(
        document
    )

    print(
        f"Detected {len(sections)} hierarchical sections"
    )

    # -------------------------------------------------------------------------
    # Chunk sections
    # -------------------------------------------------------------------------

    final_chunks = []

    for section in sections:

        section_chunks = (
            chunk_section(
                section=section,
                chunker=chunker,
                max_tokens=max_tokens,
            )
        )

        final_chunks.extend(
            section_chunks
        )

    print(
        f"Created {len(final_chunks)} final chunks"
    )

    # -------------------------------------------------------------------------
    # Convert to LangChain Documents
    # -------------------------------------------------------------------------

    documents = []

    for i, chunk in enumerate(
        final_chunks
    ):

        text = chunk[
            "text"
        ].strip()

        if not text:

            continue

        headings = chunk[
            "headings"
        ]

        items = chunk[
            "items"
        ]

        # ---------------------------------------------------------------------
        # Page numbers
        # ---------------------------------------------------------------------

        page_numbers = (
            get_page_numbers_from_items(
                items
            )
        )

        # ---------------------------------------------------------------------
        # Table
        # ---------------------------------------------------------------------

        is_table = chunk[
            "is_table"
        ]

        # ---------------------------------------------------------------------
        # Parent heading prefix
        #
        # This is intentionally added to page_content.
        #
        # Therefore the embedding model receives:
        #
        #     SECTION A. DEFINITIONS
        #     1.1. Standard Definitions
        #
        #     Def. 4...
        #
        # instead of only:
        #
        #     Def. 4...
        # ---------------------------------------------------------------------

        if headings:

            heading_prefix = (
                "\n".join(
                    headings
                )
                + "\n\n"
            )

        else:

            heading_prefix = ""

        page_content = (
            heading_prefix
            + text
        )

        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------

        metadata = {

            "policy_name":
                policy_name,

            "source":
                str(pdf_path),

            "chunk_id":
                (
                    f"{policy_name.replace(' ', '_')}"
                    f"_chunk_{i:04d}"
                ),

            "page_numbers":
                page_numbers,

            "page_start":
                (
                    page_numbers[0]
                    if page_numbers
                    else None
                ),

            "page_end":
                (
                    page_numbers[-1]
                    if page_numbers
                    else None
                ),

            "section_path":
                chunk[
                    "section_path"
                ],

            "headings":
                headings,

            "heading_depth":
                len(headings),

            "is_table":
                is_table,
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents


# =============================================================================
# STEP 14 — PROCESS ALL PDFs
# =============================================================================

def process_folder(
    folder_path: str,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    max_tokens: int = MAX_TOKENS,
) -> list[Document]:
    """
    Process every PDF inside a folder.
    """

    folder = Path(
        folder_path
    )

    pdf_paths = sorted(
        folder.glob("*.pdf")
    )

    if not pdf_paths:

        raise FileNotFoundError(
            f"No PDF files found in:\n{folder}"
        )

    print(
        f"\nFound {len(pdf_paths)} PDF(s)"
    )

    all_documents = []

    for pdf_path in pdf_paths:

        print(
            "\n" + "=" * 80
        )

        print(
            f"Processing: {pdf_path.name}"
        )

        print(
            "=" * 80
        )

        documents = build_documents(
            pdf_path=str(
                pdf_path
            ),
            embedding_model_id=(
                embedding_model_id
            ),
            max_tokens=max_tokens,
        )

        all_documents.extend(
            documents
        )

    return all_documents


# =============================================================================
# STEP 15 — SAVE DOCUMENTS TO JSON
# =============================================================================

def save_documents(
    documents: list[Document],
    output_path: str,
):
    """
    Save LangChain Documents to JSON.

    Each document contains:

        - page_content
        - metadata
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = [
        {
            "page_content":
                doc.page_content,

            "metadata":
                doc.metadata,
        }
        for doc in documents
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved {len(documents)} documents to:"
    )

    print(
        output_path
    )


# =============================================================================
# STEP 16 — MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Project structure:
    #
    # insurance_rag/
    #
    # ├── data/
    # │   ├── raw/
    # │   │   ├── policy1.pdf
    # │   │   └── policy2.pdf
    # │   │
    # │   └── processed/
    # │       └── chunks.json
    # │
    # └── src/
    #     └── insurance_rag/
    #         └── ingestion/
    #             └── load_documents.py
    #
    # parents[3] = insurance_rag/
    # -------------------------------------------------------------------------

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    PDF_FOLDER = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"PDF folder:   {PDF_FOLDER}"
    )

    # -------------------------------------------------------------------------
    # Check folder
    # -------------------------------------------------------------------------

    if not PDF_FOLDER.exists():

        raise FileNotFoundError(
            f"\nPDF folder does not exist:\n"
            f"{PDF_FOLDER}"
        )

    # -------------------------------------------------------------------------
    # Process PDFs
    # -------------------------------------------------------------------------

    docs = process_folder(
        folder_path=str(
            PDF_FOLDER
        ),
        max_tokens=MAX_TOKENS,
    )

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------

    OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "chunks.json"
    )

    save_documents(
        documents=docs,
        output_path=str(
            OUTPUT_FILE
        ),
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        f"Total chunks: {len(docs)}"
    )

    print(
        "=" * 80
    )

    # -------------------------------------------------------------------------
    # Chunks per policy
    # -------------------------------------------------------------------------

    by_policy = Counter(
        doc.metadata[
            "policy_name"
        ]
        for doc in docs
    )

    for policy, count in (
        by_policy.items()
    ):

        print(
            f"{policy}: {count} chunks"
        )

    # -------------------------------------------------------------------------
    # Show first 10 chunks
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "SAMPLE CHUNKS"
    )

    print(
        "=" * 80
    )

    for doc in docs[:10]:

        print(
            "\n" + "-" * 80
        )

        print(
            f"CHUNK ID: "
            f"{doc.metadata['chunk_id']}"
        )

        print(
            f"PAGES: "
            f"{doc.metadata['page_numbers']}"
        )

        print(
            f"SECTION: "
            f"{doc.metadata['section_path']}"
        )

        print(
            f"HEADING DEPTH: "
            f"{doc.metadata['heading_depth']}"
        )

        print(
            f"TABLE: "
            f"{doc.metadata['is_table']}"
        )

        print(
            "\nTEXT:"
        )

        print(
            doc.page_content[:1500]
        )