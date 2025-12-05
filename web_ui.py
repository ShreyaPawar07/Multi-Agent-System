import streamlit as st


st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="centered",
)

# Simple custom styling to add more color
st.markdown(
    """
    <style>
        .main {
            background: #fffbf5; /* light peach body background */
            color: #1f2933;
        }
        section[data-testid="stSidebar"] {
            background-color: #fed7aa; /* light orange background for sidebar */
            border-right: 1px solid #f97316;
        }
        /* Slider track & thumb styling (tuned for light orange sidebar) */
        .stSlider [data-baseweb="slider"] > div {
            background-color: #fffbeb; /* light track background */
        }
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: #22c55e; /* bright thumb color */
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.4);
        }
        .stSlider [data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, #22c55e, #4ade80);
        }
        .summary-box {
            background-color: #020617;
            border-radius: 0.75rem;
            padding: 1.25rem 1.5rem;
            border: 1px solid #1f2937;
        }
        .stTextArea textarea {
            background-color: #f3f4f6 !important; /* light gray background */
            color: #111827 !important;
            border-radius: 0.75rem !important;
            border: 2px solid #ec4899 !important; /* pink border */
        }
        /* Glowy primary button */
        .stButton button {
            background: linear-gradient(135deg, #22c55e, #4ade80);
            color: #0f172a;
            border-radius: 999px;
            border: none;
            box-shadow: 0 0 14px rgba(34, 197, 94, 0.7);
            font-weight: 600;
            transition: box-shadow 0.2s ease, transform 0.2s ease, filter 0.2s ease;
        }
        .stButton button:hover {
            box-shadow: 0 0 22px rgba(34, 197, 94, 0.95);
            transform: translateY(-1px);
            filter: brightness(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #be185d;'>✨ Text Summarizer ✨</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #9ca3af;'>Paste your content, tune the settings, and get a clean summary in seconds.</p>",
    unsafe_allow_html=True,
)

# Sidebar controls for model parameters
with st.sidebar:
    st.header("⚙️ Generation Settings")

    # Slider for max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=16,
        max_value=4096,
        value=1024,
        step=16,
        help="Controls the maximum length of the generated summary."
    )
    # `max_tokens` is the variable that holds the user's selected max token value.

    # Slider for temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Controls randomness: lower is more focused, higher is more creative."
    )
    # `temperature` is the variable that holds the user's selected temperature value.

st.markdown(
    "<h3 style='color: #38bdf8;'>🧾 Input Text</h3>",
    unsafe_allow_html=True,
)

# Text input from the user
text_to_summarize = st.text_area(
    "Enter text to summarize",
    height=200,
    placeholder="Paste or type your text here...",
)

# `text_to_summarize` is the variable that holds the user's input text.

if st.button("Summarize"):
    # Pass the text to your existing summarize function (defined elsewhere).
    # Replace `your_summarize_function` with the actual function name you have.
    # Note: We now also pass `max_tokens` and `temperature` along with the text.
    summary = your_summarize_function(
        text_to_summarize,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    st.markdown(
        "<h3 style='color: #4ade80;'>📌 Summary</h3>",
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
        st.write(summary)
        st.markdown("</div>", unsafe_allow_html=True)
