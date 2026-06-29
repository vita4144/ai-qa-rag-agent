import os
import warnings
from dotenv import load_dotenv

# Suppress LangChain community deprecation warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- RAG ARCHITECTURE IMPORTS ---
from langchain_community.document_loaders import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

def initialize_retrievers(file_path):
    # Only print this if we are running the evaluation script to keep the chat clean
    if __name__ != "__main__":
        print("Building Hybrid Search Retrievers (Chroma + BM25)...")
        
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()

    # 1. Vector Retriever (Semantic Search via Chroma)
    vector_store = Chroma.from_documents(docs, OpenAIEmbeddings())
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. BM25 Retriever (Exact Keyword Search)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # 3. Ensemble Retriever (Combines both 50/50)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] 
    )
    
    return ensemble_retriever

def setup_rag_chain(ensemble_retriever):
    system_prompt = (
        "You are an internal Support Triage Analyst for a software company. "
        "Your job is NOT to directly answer the user's question. "
        "Instead, analyze the user's input and use the retrieved historical tickets to summarize their intent. "
        "Always respond in the third person, describing what the user wants (e.g., 'The customer is inquiring about...'). "
        "If the retrieved context is entirely unrelated to the user's input, explicitly state that the intent cannot be classified.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, ensemble_retriever

def run_interactive_chat():
    print("==================================================")
    print("🤖 AI Customer Support Agent Initialized.")
    print("Type your question below. Type 'exit' or 'quit' to stop.")
    print("==================================================")
    
    # Initialize the architecture
    ensemble_retriever = initialize_retrievers("support_tickets.csv")
    rag_chain, _ = setup_rag_chain(ensemble_retriever)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down Support Agent...")
            break
            
        if not user_input.strip():
            continue
            
        try:
            # Stream or invoke the response
            response = rag_chain.invoke(user_input)
            print(f"\n🤖 Agent: {response}")
        except Exception as e:
            print(f"\n⚠️ System Error: {e}")

if __name__ == "__main__":
    run_interactive_chat()
