import sys
import re
import json
import subprocess
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEXES_DIR = PROJECT_ROOT / "indexes"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import RAGPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResearchPaper RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GENERAL
       ====================================================== */

    .stApp {
        background-color: #f8fafc;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .sidebar-subtitle {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 6px 11px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #22c55e;
        border-radius: 50%;
    }

    .document-title {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .document-info {
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 6px;
        line-height: 1.4;
    }


    /* ======================================================
       MAIN HEADER
       ====================================================== */

    .page-title {
        font-size: 2.2rem;
        font-weight: 750;
        color: #111827;
        margin-bottom: 0.15rem;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }


    /* ======================================================
       SUGGESTED QUESTIONS
       ====================================================== */

    .section-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 0.65rem;
    }


    /* ======================================================
       USER QUESTION
       ====================================================== */

    .user-question {
        background: #eef2ff;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 22px 0 10px auto;
        max-width: 80%;
        color: #1f2937;
        line-height: 1.55;
    }


    /* ======================================================
       ANSWER
       ====================================================== */

    .answer-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 8px 0 18px 0;
    }

    .answer-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #4f46e5;
        margin-bottom: 7px;
    }


    /* ======================================================
       EVIDENCE
       ====================================================== */

    .evidence-heading {
        font-size: 0.9rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    .evidence-meta {
        font-size: 0.78rem;
        color: #6b7280;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

PAPERS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INDEXES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD RAG PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():
    return RAGPipeline()


with st.spinner("Loading research assistant..."):
    pipeline = load_pipeline()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = "All Papers"

if "upload_message" not in st.session_state:
    st.session_state.upload_message = None


# ============================================================
# GET INDEXED PAPERS
# ============================================================

def get_indexed_papers():
    """
    Read the current chunk metadata and return unique
    research paper filenames.
    """

    chunks_file = INDEXES_DIR / "chunks.jsonl"

    if not chunks_file.exists():
        return []

    papers = set()

    try:

        with open(
            chunks_file,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    item = json.loads(line)

                    filename = item.get(
                        "filename"
                    )

                    if filename:
                        papers.add(filename)

                except json.JSONDecodeError:
                    continue

    except Exception:
        return []

    return sorted(
        papers,
        key=str.lower,
    )


# ============================================================
# TEXT CLEANING FUNCTIONS
# ============================================================

def clean_source_text(text):
    """
    Clean text extracted from the PDF so that
    raw HTML/Markdown does not appear in the UI.
    """

    text = str(text).strip()

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    # Decode common HTML entities
    text = (
        text
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    # Remove Markdown headings
    text = re.sub(
        r"#{1,6}\s*",
        "",
        text,
    )

    # Remove Markdown emphasis
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

    # Clean excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def create_excerpt(
    text,
    max_chars=400,
):
    """
    Create a short readable excerpt from
    the retrieved source passage.
    """

    text = clean_source_text(text)

    if len(text) <= max_chars:
        return text

    excerpt = text[:max_chars]

    # Avoid cutting in the middle of a word
    if " " in excerpt:
        excerpt = excerpt.rsplit(
            " ",
            1,
        )[0]

    return excerpt + "..."


# ============================================================
# DISPLAY NAME
# ============================================================

def display_paper_name(filename):
    """
    Convert a PDF filename into a cleaner UI label.
    """

    name = Path(filename).stem

    name = name.replace(
        "_",
        " ",
    )

    name = name.replace(
        "-",
        " ",
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    ).strip()

    return name


# ============================================================
# REBUILD INDEX AFTER UPLOAD
# ============================================================

def rebuild_indexes():
    """
    Run the existing PDF processor and embedding script.

    The processor processes all PDFs in data/papers.
    The embedding script rebuilds the combined FAISS index.
    """

    python_executable = sys.executable

    processor_script = (
        SRC_DIR / "pdf_processor.py"
    )

    embeddings_script = (
        SRC_DIR / "embeddings.py"
    )

    if not processor_script.exists():
        raise FileNotFoundError(
            "src/pdf_processor.py was not found."
        )

    if not embeddings_script.exists():
        raise FileNotFoundError(
            "src/embeddings.py was not found."
        )

    # --------------------------------------------------------
    # Process PDFs
    # --------------------------------------------------------

    processor_result = subprocess.run(
        [
            python_executable,
            str(processor_script),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if processor_result.returncode != 0:

        error_message = (
            processor_result.stderr.strip()
            or processor_result.stdout.strip()
            or "PDF processing failed."
        )

        raise RuntimeError(
            error_message
        )

    # --------------------------------------------------------
    # Generate embeddings and rebuild FAISS
    # --------------------------------------------------------

    embedding_result = subprocess.run(
        [
            python_executable,
            str(embeddings_script),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if embedding_result.returncode != 0:

        error_message = (
            embedding_result.stderr.strip()
            or embedding_result.stdout.strip()
            or "Embedding generation failed."
        )

        raise RuntimeError(
            error_message
        )

    return embedding_result.stdout


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">'
        '📚 ResearchPaper RAG'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Evidence-Based Research Assistant'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status">
            <span class="status-dot"></span>
            System Ready
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PAPER SELECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="document-title">'
        'Research Papers'
        '</div>',
        unsafe_allow_html=True,
    )

    indexed_papers = get_indexed_papers()

    paper_options = [
        "All Papers"
    ] + indexed_papers

    current_selection = (
        st.session_state.selected_paper
    )

    if current_selection not in paper_options:
        current_selection = "All Papers"

    selected_paper = st.selectbox(
        "Select paper",
        paper_options,
        index=paper_options.index(
            current_selection
        ),
        format_func=lambda x: (
            "All Papers"
            if x == "All Papers"
            else display_paper_name(x)
        ),
        label_visibility="collapsed",
    )

    st.session_state.selected_paper = (
        selected_paper
    )

    # --------------------------------------------------------
    # PAPER COUNT
    # --------------------------------------------------------

    if indexed_papers:

        paper_count = len(
            indexed_papers
        )

        if paper_count == 1:
            paper_text = "1 paper available"
        else:
            paper_text = (
                f"{paper_count} papers available"
            )

        st.markdown(
            f'<div class="document-info">'
            f'{paper_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="document-info">'
            'No papers indexed yet.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # ADD RESEARCH PAPER
    # --------------------------------------------------------

    st.markdown(
        '<div class="document-title">'
        'Add Research Paper'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF research paper",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:

        upload_clicked = st.button(
            "Add Research Paper",
            use_container_width=True,
        )

        if upload_clicked:

            destination = (
                PAPERS_DIR /
                uploaded_file.name
            )

            try:

                # ------------------------------------------------
                # Check duplicate
                # ------------------------------------------------

                if destination.exists():

                    st.warning(
                        "This paper is already available."
                    )

                else:

                    # ------------------------------------------------
                    # Save uploaded PDF
                    # ------------------------------------------------

                    with open(
                        destination,
                        "wb",
                    ) as file:

                        file.write(
                            uploaded_file.getbuffer()
                        )

                    # ------------------------------------------------
                    # Process and rebuild index
                    # ------------------------------------------------

                    with st.spinner(
                        "Adding the research paper..."
                    ):

                        rebuild_indexes()

                    # ------------------------------------------------
                    # Clear cached pipeline
                    # ------------------------------------------------

                    load_pipeline.clear()

                    # ------------------------------------------------
                    # Clear previous conversation
                    # ------------------------------------------------

                    st.session_state.messages = []
                    st.session_state.pending_question = None

                    # ------------------------------------------------
                    # Select new paper
                    # ------------------------------------------------

                    st.session_state.selected_paper = (
                        uploaded_file.name
                    )

                    st.session_state.upload_message = (
                        f'"{display_paper_name(uploaded_file.name)}" '
                        "was added successfully."
                    )

                    st.rerun()

            except Exception as e:

                # If indexing fails, remove the PDF that
                # was just uploaded so the library does not
                # contain an unindexed paper.

                try:

                    if destination.exists():
                        destination.unlink()

                except Exception:
                    pass

                st.error(
                    "The research paper could not be added."
                )

                st.caption(
                    f"{type(e).__name__}: {e}"
                )

    # --------------------------------------------------------
    # UPLOAD SUCCESS MESSAGE
    # --------------------------------------------------------

    if st.session_state.upload_message:

        st.success(
            st.session_state.upload_message
        )

        st.session_state.upload_message = None

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.pending_question = None

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="page-title">'
    'ResearchPaper RAG'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-subtitle">'
    'Ask questions across your research library and explore '
    'answers with supporting evidence.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# CURRENT PAPER INFORMATION
# ============================================================

if selected_paper == "All Papers":

    st.caption(
        f"Searching across {len(indexed_papers)} "
        f"research paper"
        f"{'s' if len(indexed_papers) != 1 else ''}."
    )

else:

    st.caption(
        "Currently viewing: "
        f"{display_paper_name(selected_paper)}"
    )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.markdown(
        '<div class="section-label">'
        'Suggested questions'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "What is the main contribution?",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What is the main contribution?"
            )

            st.rerun()

    with col2:

        if st.button(
            "What dataset was used?",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What dataset was used?"
            )

            st.rerun()

    col3, col4 = st.columns(2)

    with col3:

        if st.button(
            "What are the key results?",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What are the key results?"
            )

            st.rerun()

    with col4:

        if st.button(
            "What are the limitations?",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What are the limitations?"
            )

            st.rerun()


# ============================================================
# DISPLAY CONVERSATION
# ============================================================

for message in st.session_state.messages:

    question = message["question"]
    answer = message["answer"]
    sources = message.get(
        "sources",
        [],
    )

    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="user-question">
            {question}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    st.markdown(
        '<div class="answer-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="answer-label">'
        'Answer'
        '</div>',
        unsafe_allow_html=True,
    )

    # Render answer through Streamlit
    # rather than injecting it into HTML.
    st.markdown(answer)

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SUPPORTING EVIDENCE
    # --------------------------------------------------------

    if sources:

        with st.expander(
            f"Supporting Evidence · "
            f"{len(sources)} sources"
        ):

            for i, source in enumerate(
                sources,
                start=1,
            ):

                # --------------------------------------------
                # Source section
                # --------------------------------------------

                section = clean_source_text(
                    source.get(
                        "section",
                        "",
                    )
                )

                if not section:
                    section = "Research Paper"

                # --------------------------------------------
                # Source page
                # --------------------------------------------

                page = source.get(
                    "page",
                    "",
                )

                # --------------------------------------------
                # Source paper
                # --------------------------------------------

                filename = source.get(
                    "filename",
                    "",
                )

                paper_name = (
                    display_paper_name(
                        filename
                    )
                    if filename
                    else "Research Paper"
                )

                # --------------------------------------------
                # Evidence text
                # --------------------------------------------

                raw_text = source.get(
                    "text",
                    "",
                )

                excerpt = create_excerpt(
                    raw_text,
                    max_chars=400,
                )

                # --------------------------------------------
                # Source heading
                # --------------------------------------------

                st.markdown(
                    f"**[S{i}] {paper_name}**"
                )

                # --------------------------------------------
                # Source metadata
                # --------------------------------------------

                metadata = []

                if page:
                    metadata.append(
                        f"Page {page}"
                    )

                if section:
                    metadata.append(
                        section
                    )

                metadata_text = " · ".join(
                    metadata
                )

                if metadata_text:

                    st.caption(
                        metadata_text
                    )

                # --------------------------------------------
                # Source excerpt
                # --------------------------------------------

                st.write(
                    excerpt
                )

                # --------------------------------------------
                # Separator
                # --------------------------------------------

                if i < len(sources):
                    st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

pending_question = (
    st.session_state.pending_question
)

st.session_state.pending_question = None

question = st.chat_input(
    "Ask a question about your research papers..."
)

if pending_question:
    question = pending_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if question:

        # ----------------------------------------------------
        # Determine paper filter
        # ----------------------------------------------------

        if selected_paper == "All Papers":
            filename_filter = None
        else:
            filename_filter = selected_paper

        with st.spinner(
            "Finding the answer..."
        ):

            try:

                # ------------------------------------------------
                # IMPORTANT:
                #
                # The pipeline now performs:
                #
                # retrieval
                #     ↓
                # reranking
                #     ↓
                # answer generation
                #
                # and returns the EXACT SAME sources that
                # were supplied to the LLM.
                # ------------------------------------------------

                result = (
                    pipeline.answer_question_with_sources(
                        question=question,
                        candidate_k=10,
                        top_k=6,
                        filename=filename_filter,
                    )
                )

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                sources = result.get(
                    "sources",
                    [],
                )

                # ------------------------------------------------
                # Store conversation
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    }
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "Sorry, I couldn't generate an answer "
                    "right now. Please try again."
                )

                st.caption(
                    f"{type(e).__name__}: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ResearchPaper RAG · Evidence-grounded research assistant
    </div>
    """,
    unsafe_allow_html=True,
)