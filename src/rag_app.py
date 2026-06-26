import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def initialize_vector_db(file_path):
    loader = CSVLoader(file_path=file_path, source_column="Ticket_Description")
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings)
    return vector_store

def setup_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system_prompt = (
        "You are a reliable customer support QA assistant. Use the following pieces of "
        "retrieved context to answer the user's question accurately. If the context does not contain "
        "the answer, explicitly state that you do not know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def query_app(rag_chain, question):
    response = rag_chain.invoke({"input": question})
    contexts = [doc.page_content for doc in response["context"]]
    return {
        "answer": response["answer"],
        "contexts": contexts
    }

# Global cache variable to keep the chat loop fast and efficient
_chain_instance = None

def query_rag_system(user_query, vector_db):
    """
    Bridges the interactive loop with your LangChain retrieval infrastructure.
    Caches the chain layout so it does not rebuild on every single keystroke.
    """
    global _chain_instance
    if _chain_instance is None:
        # Generates your classic retrieval chain layout using your existing setup function
        _chain_instance = setup_rag_chain(vector_db)
    
    # Run the user input through the active pipeline
    raw_response = _chain_instance.invoke({"input": user_query})
    
    # Normalize keys so your loop's 'contexts' check never throws a KeyError
    return {
        "answer": raw_response.get("answer", "No response generated."),
        "contexts": raw_response.get("context", raw_response.get("contexts", []))
    }

def run_interactive_chat():
    print("==================================================")
    print("🤖 AI Customer Support Agent Initialized.")
    print("Type your question below. Type 'exit' or 'quit' to stop.")
    print("==================================================")
    
    # Initialize your vector database using the correct literal file string
    vector_db = initialize_vector_db("support_tickets.csv") 
    
    while True:
        user_query = input("\n👤 You: ")
        
        # Check if the user wants to exit
        if user_query.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_query.strip():
            continue
            
        try:
            # Query your stable RAG application pipeline
            response = query_rag_system(user_query, vector_db)
            
            print(f"\n🤖 Agent: {response['answer']}")
            
            # Print the source tickets for verification
            print("\n📌 [QA Source Context Used]:")
            for i, doc in enumerate(response['contexts'], 1):
                print(f"  Snippet {i}: {doc.page_content[:120]}...")
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_interactive_chat()