from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END

from rag.retriever import retriever

from rag.prompts import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
    isuse_prompt,
    rewrite_for_retrieval_prompt,
)

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=MISTRAL_API_KEY,
)

###########################################################
# Graph State
###########################################################

class State(TypedDict):
    question: str

    retrieval_query: str
    rewrite_tries: int

    need_retrieval: bool

    docs: List[Document]
    relevant_docs: List[Document]

    context: str
    answer: str

    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support",
    ]

    evidence: List[str]

    retries: int

    isuse: Literal[
        "useful",
        "not_useful",
    ]

    use_reason: str


###########################################################
# Structured Output Models
###########################################################

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed."
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="Whether document is relevant."
    )


class IsSUPDecision(BaseModel):
    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support",
    ]

    evidence: List[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: Literal[
        "useful",
        "not_useful",
    ]

    reason: str


class RewriteDecision(BaseModel):
    retrieval_query: str


###########################################################
# Structured LLMs
###########################################################

should_retrieve_llm = llm.with_structured_output(
    RetrieveDecision
)

relevance_llm = llm.with_structured_output(
    RelevanceDecision
)

issup_llm = llm.with_structured_output(
    IsSUPDecision
)

isuse_llm = llm.with_structured_output(
    IsUSEDecision
)

rewrite_llm = llm.with_structured_output(
    RewriteDecision
)

###########################################################
# Node 1
# Decide Retrieval
###########################################################

def decide_retrieval(state: State):

    decision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(
            question=state["question"]
        )
    )

    return {
        "need_retrieval": decision.should_retrieve
    }


def route_after_decide(
    state: State,
) -> Literal["generate_direct", "retrieve"]:

    if state["need_retrieval"]:
        return "retrieve"

    return "generate_direct"


###########################################################
# Node 2
# Direct Generation
###########################################################

def generate_direct(state: State):

    output = llm.invoke(

        direct_generation_prompt.format_messages(
            question=state["question"]
        )

    )

    return {
        "answer": output.content
    }


###########################################################
# Node 3
# Retrieve
###########################################################

def retrieve(state: State):

    query = (
        state.get("retrieval_query")
        or state["question"]
    )

    docs = retriever.invoke(query)

    return {
        "docs": docs
    }


###########################################################
# Node 4
# Relevance Filter
###########################################################

def is_relevant(state: State):

    relevant_docs = []

    for doc in state.get("docs", []):

        decision = relevance_llm.invoke(

            is_relevant_prompt.format_messages(

                question=state["question"],
                document=doc.page_content,

            )

        )

        if decision.is_relevant:
            relevant_docs.append(doc)

    return {

        "relevant_docs": relevant_docs

    }


def route_after_relevance(
    state: State,
) -> Literal[
    "generate_from_context",
    "no_answer_found",
]:

    if len(state.get("relevant_docs", [])) > 0:

        return "generate_from_context"

    return "no_answer_found"


###########################################################
# Node 5
# Generate From Context
###########################################################

def generate_from_context(state: State):

    context = "\n\n---\n\n".join(

        doc.page_content

        for doc in state.get(
            "relevant_docs",
            [],
        )

    )

    context = context.strip()

    if not context:

        return {

            "answer": "No answer found.",

            "context": "",

        }

    output = llm.invoke(

        rag_generation_prompt.format_messages(

            question=state["question"],

            context=context,

        )

    )

    return {

        "answer": output.content,

        "context": context,

    }


def no_answer_found(state: State):

    return {

        "answer": "No answer found.",

        "context": "",

    }
    
    
    
    
    
###########################################################
# Constants
###########################################################

MAX_RETRIES = 10
MAX_REWRITE_TRIES = 3


###########################################################
# Node 6
# IsSUP
###########################################################

def is_sup(state: State):

    decision = issup_llm.invoke(

        issup_prompt.format_messages(

            question=state["question"],

            answer=state.get("answer", ""),

            context=state.get("context", ""),

        )

    )

    return {

        "issup": decision.issup,

        "evidence": decision.evidence,

    }


def route_after_issup(
    state: State,
) -> Literal["accept_answer", "revise_answer"]:

    if state.get("issup") == "fully_supported":

        return "accept_answer"

    if state.get("retries", 0) >= MAX_RETRIES:

        return "accept_answer"

    return "revise_answer"


def accept_answer(state: State):

    return {}


###########################################################
# Node 7
# Revise Answer
###########################################################

def revise_answer(state: State):

    output = llm.invoke(

        revise_prompt.format_messages(

            question=state["question"],

            answer=state.get("answer", ""),

            context=state.get("context", ""),

        )

    )

    return {

        "answer": output.content,

        "retries": state.get("retries", 0) + 1,

    }


###########################################################
# Node 8
# IsUSE
###########################################################

def is_use(state: State):

    decision = isuse_llm.invoke(

        isuse_prompt.format_messages(

            question=state["question"],

            answer=state.get("answer", ""),

        )

    )

    return {

        "isuse": decision.isuse,

        "use_reason": decision.reason,

    }


def route_after_isuse(
    state: State,
) -> Literal[
    "END",
    "rewrite_question",
    "no_answer_found",
]:

    if state.get("isuse") == "useful":

        return "END"

    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:

        return "no_answer_found"

    return "rewrite_question"


###########################################################
# Node 9
# Rewrite Retrieval Query
###########################################################

def rewrite_question(state: State):

    decision = rewrite_llm.invoke(

        rewrite_for_retrieval_prompt.format_messages(

            question=state["question"],

            retrieval_query=state.get(
                "retrieval_query",
                "",
            ),

            answer=state.get(
                "answer",
                "",
            ),

        )

    )

    return {

        "retrieval_query": decision.retrieval_query,

        "rewrite_tries": state.get(
            "rewrite_tries",
            0,
        ) + 1,

        "docs": [],

        "relevant_docs": [],

        "context": "",

    }


###########################################################
# Build Graph
###########################################################

graph = StateGraph(State)


###########################################################
# Register Nodes
###########################################################

graph.add_node(
    "decide_retrieval",
    decide_retrieval,
)

graph.add_node(
    "generate_direct",
    generate_direct,
)

graph.add_node(
    "retrieve",
    retrieve,
)

graph.add_node(
    "is_relevant",
    is_relevant,
)

graph.add_node(
    "generate_from_context",
    generate_from_context,
)

graph.add_node(
    "no_answer_found",
    no_answer_found,
)

graph.add_node(
    "is_sup",
    is_sup,
)

graph.add_node(
    "revise_answer",
    revise_answer,
)

graph.add_node(
    "is_use",
    is_use,
)

graph.add_node(
    "rewrite_question",
    rewrite_question,
)


###########################################################
# Graph Edges
###########################################################

graph.add_edge(
    START,
    "decide_retrieval",
)

graph.add_conditional_edges(

    "decide_retrieval",

    route_after_decide,

    {

        "generate_direct": "generate_direct",

        "retrieve": "retrieve",

    },

)

graph.add_edge(
    "generate_direct",
    END,
)

graph.add_edge(
    "retrieve",
    "is_relevant",
)

graph.add_conditional_edges(

    "is_relevant",

    route_after_relevance,

    {

        "generate_from_context":
            "generate_from_context",

        "no_answer_found":
            "no_answer_found",

    },

)

graph.add_edge(
    "generate_from_context",
    "is_sup",
)

graph.add_conditional_edges(

    "is_sup",

    route_after_issup,

    {

        "accept_answer":
            "is_use",

        "revise_answer":
            "revise_answer",

    },

)

graph.add_edge(
    "revise_answer",
    "is_sup",
)

graph.add_conditional_edges(

    "is_use",

    route_after_isuse,

    {

        "END":
            END,

        "rewrite_question":
            "rewrite_question",

        "no_answer_found":
            "no_answer_found",

    },

)

graph.add_edge(
    "rewrite_question",
    "retrieve",
)

graph.add_edge(
    "no_answer_found",
    END)


###########################################################
# Compile
###########################################################

app = graph.compile()






###########################################################
# Public API
###########################################################

def ask(question: str) -> dict:
    """
    Runs the complete Self-RAG workflow for a single question.

    Parameters
    ----------
    question : str
        User question

    Returns
    -------
    dict
        Final graph state containing:
        - answer
        - context
        - issup
        - evidence
        - isuse
        - use_reason
        - retries
        - rewrite_tries
        etc.
    """

    initial_state = {

        "question": question,

        "retrieval_query": "",

        "rewrite_tries": 0,

        "need_retrieval": False,

        "docs": [],

        "relevant_docs": [],

        "context": "",

        "answer": "",

        "issup": "no_support",

        "evidence": [],

        "retries": 0,

        "isuse": "not_useful",

        "use_reason": "",

    }

    result = app.invoke(

        initial_state,

        config={

            "recursion_limit": 80

        },

    )

    return result


###########################################################
# Run Locally
###########################################################

if __name__ == "__main__":

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        result = ask(question)

        print("\n" + "=" * 70)

        print("\nAnswer:\n")

        print(result["answer"])

        print("\n" + "=" * 70)

        print(f"Useful      : {result['isuse']}")

        print(f"Faithfulness: {result['issup']}")

        print(f"Retries     : {result['retries']}")

        print(f"Rewrites    : {result['rewrite_tries']}")
        
        
        
        
        
        
        