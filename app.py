
import html
import urllib.request
from pathlib import Path

import sentencepiece as spm
import streamlit as st
from tokenizers import Tokenizer

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tokenizer Playground",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Tokenizer Playground")
st.markdown(
    "Explore how different tokenizers break the same text into tokens, "
    "assign IDs, and reconstruct the original text."
)

BASE_DIR = Path(__file__).parent

SP_MODEL_PATH = BASE_DIR / "tokenizer.model"
BPE_MODEL_PATH = BASE_DIR / "gpt2-tokenizer.json"

SP_MODEL_URL = (
    "https://huggingface.co/google/mt5-small/resolve/main/"
    "spiece.model"
)

BPE_MODEL_URL = (
    "https://huggingface.co/gpt2/resolve/main/tokenizer.json"
)

COLORS = [
    "#DCEBFF",
    "#DDF5E5",
    "#FFE8CC",
    "#F2DEFF",
    "#FFE0E5",
    "#D9F4F4",
]


# --------------------------------------------------
# Load pretrained tokenizers
# --------------------------------------------------

@st.cache_resource
def load_unigram():
    if not SP_MODEL_PATH.exists():
        urllib.request.urlretrieve(
            SP_MODEL_URL,
            str(SP_MODEL_PATH),
        )

    processor = spm.SentencePieceProcessor()
    processor.load(str(SP_MODEL_PATH))
    return processor


@st.cache_resource
def load_bpe():
    if not BPE_MODEL_PATH.exists():
        urllib.request.urlretrieve(
            BPE_MODEL_URL,
            str(BPE_MODEL_PATH),
        )

    return Tokenizer.from_file(str(BPE_MODEL_PATH))


# --------------------------------------------------
# Display colored tokens safely
# --------------------------------------------------

def show_colored_tokens(tokens):
    blocks = []

    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        safe_token = html.escape(token)

        blocks.append(
            f'<span style="display:inline-block;'
            f'background-color:{color};color:#172033;'
            f'padding:8px 10px;margin:4px;'
            f'border-radius:8px;font-family:monospace;'
            f'font-size:14px;overflow-wrap:anywhere;">'
            f'{safe_token}</span>'
        )

    st.markdown(
        "".join(blocks),
        unsafe_allow_html=True,
    )


def show_token_details(tokens, ids):
    st.markdown("**Token IDs**")
    st.code(str(ids))

    st.markdown("**Token-by-token breakdown**")
    st.dataframe(
        [
            {
                "Position": i + 1,
                "Token": token,
                "Token ID": token_id,
            }
            for i, (token, token_id)
            in enumerate(zip(tokens, ids))
        ],
        use_container_width=True,
        hide_index=True,
    )


def show_decoding_result(decoded, original):
    st.markdown("**Decoded text**")
    st.code(decoded if decoded else "(empty string)")

    if decoded == original:
        st.success("Decoded text matches the original input exactly.")
    else:
        st.warning(
            "Decoded text differs from the original input. "
            "This may result from tokenizer normalization or "
            "how the tokenizer handles special tokens."
        )


# --------------------------------------------------
# Main application
# --------------------------------------------------

try:
    with st.spinner("Loading pretrained tokenizers..."):
        unigram = load_unigram()
        bpe = load_bpe()

    st.success("Both tokenizers loaded successfully!")

    text = st.text_area(
        "Enter text to compare",
        value="anthropomorphize",
        height=120,
        placeholder="Try English, Hindi, numbers, punctuation, or emoji.",
    )

    st.caption(
        "Tip: Try 'anthropomorphize', 'Hello 😊', or 'नमस्ते दुनिया'."
    )

    if st.button("Compare tokenizers", type="primary"):
        if not text:
            st.warning("Please enter some text.")
        else:
            # SentencePiece Unigram
            uni_tokens = unigram.encode(text, out_type=str)
            uni_ids = unigram.encode(text, out_type=int)
            uni_decoded = unigram.decode(uni_ids)

            # GPT-2 byte-level BPE
            bpe_encoding = bpe.encode(text)
            bpe_tokens = bpe_encoding.tokens
            bpe_ids = bpe_encoding.ids

            # Decode IDs using the tokenizer's byte-level decoder.
            # Do not decode the visible token strings directly.
            bpe_decoded = bpe.decode(bpe_ids)

            # Summary
            st.subheader("📊 Comparison overview")

            col1, col2, col3 = st.columns(3)
            col1.metric("Input characters", len(text))
            col2.metric("SentencePiece tokens", len(uni_tokens))
            col3.metric("GPT-2 BPE tokens", len(bpe_tokens))

            difference = len(uni_tokens) - len(bpe_tokens)

            if difference > 0:
                st.info(
                    f"SentencePiece produced {difference} more token(s) "
                    "for this input."
                )
            elif difference < 0:
                st.info(
                    f"GPT-2 BPE produced {abs(difference)} more token(s) "
                    "for this input."
                )
            else:
                st.info(
                    "Both tokenizers produced the same number of tokens."
                )

            # Side-by-side results
            left, right = st.columns(2)

            with left:
                st.markdown("### 🟢 SentencePiece Unigram")
                st.caption("Pretrained mT5-small tokenizer")

                show_colored_tokens(uni_tokens)
                show_token_details(uni_tokens, uni_ids)
                show_decoding_result(uni_decoded, text)

            with right:
                st.markdown("### 🔵 GPT-2 BPE")
                st.caption("Pretrained GPT-2 byte-level BPE tokenizer")

                show_colored_tokens(bpe_tokens)
                show_token_details(bpe_tokens, bpe_ids)
                show_decoding_result(bpe_decoded, text)

            # Educational explanation
            with st.expander("🔎 Why are the results different?"):
                st.markdown(
                    """
                    **SentencePiece Unigram**
                    - Uses a learned vocabulary and a probabilistic
                      segmentation approach.
                    - May represent a word using several subword pieces.

                    **GPT-2 BPE**
                    - Uses byte-level input representation and learned
                      merge rules to construct tokens.
                    - Non-English text may produce more tokens, and the
                      visible token strings can look unusual.

                    **Important:** These are two different pretrained
                    models with different vocabularies, training data,
                    and normalization behavior. This is an illustration
                    of their outputs, not a controlled experiment that
                    isolates the algorithm alone.

                    A lower token count does not automatically mean a
                    tokenizer is better.
                    """
                )

except Exception as e:
    st.error(f"Could not load or run the tokenizers: {e}")
    st.info(
        "Check your internet connection and verify that the model "
        "files are valid."
    )
