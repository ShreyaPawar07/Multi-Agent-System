import os

from dotenv import load_dotenv
import re
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from vectordb import query_vector_db  # your own file

load_dotenv()

# -----------------------------------------------------------------------------
# LLM CONFIG
# -----------------------------------------------------------------------------
llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast:free",          # <-- adjust if needed
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),    # None is fine if using direct OpenAI/xAI
    temperature=0.3,
)

# -----------------------------------------------------------------------------
# TOOL: policy_lookup  (uses your vector DB)
# -----------------------------------------------------------------------------
@tool
def policy_lookup(query: str) -> str:
    """
    Fetch up to five passages from the policy vector store.
    The query is a natural language question from the user.
    Returns "NO_MATCH" if nothing is found.
    """
    passages = list(query_vector_db(query, k=3))
    if not passages:
        return "NO_MATCH"

    return "\n\n".join(
        f"[Passage {idx}] {text.strip()}"
        for idx, text in enumerate(passages, start=1)
    )


# -----------------------------------------------------------------------------
# AGENT 1: Retrieval agent (LLM + tool in a loop via create_agent)
# We use LangChain v1's create_agent, which returns a LangGraph-based agent.
# It expects "messages": [...] as input and returns {"messages": [...] }.
# -----------------------------------------------------------------------------
retrieval_agent = create_agent(
    model=llm,
    tools=[policy_lookup],
    system_prompt=(
        "You are a retrieval agent for company policies.\n"
        "- Given the user's question, you MUST call the `policy_lookup` tool "
        "exactly once using the FULL user question as the query.\n"
        "- After getting the tool result, you MUST return ONLY the raw tool "
        "output as your final answer (no extra commentary, no rephrasing).\n"
    ),
)

def _clean_query(text: str) -> str:
    """Normalize user query: lowercase, remove punctuation, collapse spaces."""
    lowered = text.lower()
    # Remove punctuation / non-word characters, keep letters, numbers, whitespace
    no_punct = re.sub(r"[^\w\s]", " ", lowered)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", no_punct).strip()

def run_retrieval_agent(user_query: str) -> str:
    """
    Call the retrieval agent with the user question and return the raw passages
    text (or 'NO_MATCH').

    This is Agent 1 in the multi-agent setup.
    """
    cleaned_query = _clean_query(user_query)
    result = retrieval_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": cleaned_query}
            ]
        }
    )
    # create_agent returns a state dict with a "messages" list.
    messages = result["messages"]
    return messages[-1].content  # last assistant message content


# -----------------------------------------------------------------------------
# AGENT 2: Summary / Explanation agent (LLM chain using ChatPromptTemplate)
# This agent takes the retrieved passages and rewrites them into a
# human-friendly answer.
# -----------------------------------------------------------------------------

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a policy explainer assistant.\n"
            "You summarize policy snippets into clear, human-friendly answers.\n"
            "- Use only the information in the retrieved passages.\n"
            "- Prefer giving a direct, affirmative answer when the passages contain\n"
            "  at least some relevant details.\n"
            "- Avoid over-stating that 'no details are provided' if there is any\n"
            "  partial or contextual information in the passages.\n"
            "- Only say that the passages do not answer the question when they\n"
            "  contain no relevant policy content at all.",
        ),
        (
            "human",
            "Employee question:\n{question}\n\n"
            "Retrieved passages:\n{passages}",
        ),
    ]
)

summary_chain = summary_prompt | llm | StrOutputParser()


def run_summary_agent(question: str, passages: str) -> str:
    """
    Use the summary agent to turn raw policy passages into a human-friendly answer.

    This is Agent 2 in the multi-agent setup.
    """
    if passages.strip() == "NO_MATCH":
        return "I could not find any policy text that answers that question."

    return summary_chain.invoke(
        {
            "question": question,
            "passages": passages,
        }
    )


# -----------------------------------------------------------------------------
# ORCHESTRATOR: end-to-end helper
# -----------------------------------------------------------------------------
def answer_question(question: str) -> str:
    """
    High-level function to:
    1) Ask the retrieval agent for relevant policy passages.
    2) Pass those passages to the summary agent.
    3) Return the final human-friendly answer.

    Memory / chat history can be handled in the Streamlit layer by:
      - keeping a history list and
      - optionally augmenting `question` with previous context before calling this.
    """
    passages = run_retrieval_agent(question)
    # print(passages)
    return run_summary_agent(question, passages)


if __name__ == "__main__":
    user_question = "What is the policy on vacation?"
    print(answer_question(user_question))
