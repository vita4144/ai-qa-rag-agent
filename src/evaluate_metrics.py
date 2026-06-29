import sys
import types
import warnings

# Suppress harmless deprecation warnings from Ragas to keep the console clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- CRITICAL RAGAS FIX (MONKEY PATCH) ---
dummy_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertexai
# -----------------------------------------

import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

# --- UPDATED IMPORTS FOR RAGAS v0.2+ ---
# Metrics are now Classes that must be explicitly instantiated
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# --------------------------------------

from rag_app import initialize_retrievers, setup_rag_chain

load_dotenv()

def run_evaluation():
    print("Step 1: Indexing Hybrid vector store...")
    ensemble_retriever = initialize_retrievers("support_tickets.csv")
    rag_chain, active_retriever = setup_rag_chain(ensemble_retriever)

    print("Step 2: Loading baseline test dataset...")
    df = pd.read_csv("eval_baseline.csv")
    
    results = []
    print("Step 3: Querying the live RAG system...")
    for index, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        
        print(f"   -> Processing Test Case {index + 1}/{len(df)}: '{question}'")
        
        answer = rag_chain.invoke(question)
        docs = active_retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # CRITICAL UPDATE: Mapping to the new Ragas v0.2 schema
        results.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth
        })
        
    evaluation_dataset = Dataset.from_pandas(pd.DataFrame(results))
    
    print("\nStep 4: Running RAGAS automated metrics evaluation...")
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Instantiate the metric classes with ()
    metrics = [
        ContextPrecision(), 
        Faithfulness(), 
        AnswerRelevancy()
    ]
    
    # The new evaluate function handles the internal bindings automatically
    evaluation_results = evaluate(
        dataset=evaluation_dataset, 
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    print("\n=== AGGREGATE QA SCORES ===")
    print(evaluation_results)
    
    df_results = evaluation_results.to_pandas()
    df_results.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed verification spreadsheet saved as 'evaluation_results.csv'")

if __name__ == "__main__":
    run_evaluation()
    