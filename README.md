**Automated RAG Quality Evaluation Pipeline**

🎯 Project Objective

The goal of this project is to build a Retrieval-Augmented Generation (RAG) system for customer support and to implement an automated quality assurance pipeline using the **Ragas** framework.
Rather than relying on manual spot-checking, this project demonstrates how to use LLM-as-a-judge methodologies to quantitatively measure hallucinations, retrieval accuracy, and response usefulness in production AI systems.

## 🛠️ How the Core Application Works (`rag_app.py`)

The core of this project is an AI Customer Support Agent built using the **LangChain** framework to deploy a Retrieval-Augmented Generation (RAG) pipeline. Instead of relying solely on the base knowledge of an LLM, the application actively pulls context from our verified data source before generating an answer.

The pipeline processes queries through five distinct stages:

### 1. Ingestion (Reading the Data)
* **Action:** The system connects to the target dataset (`support_tickets.csv`) using a localized data loader.
* **Targeting:** It extracts and isolates unstructured customer text specifically from the `Ticket_Description` column, preparing the raw complaints for processing.

### 2. Embedding & Storage (Semantic Mapping)
* **Action:** The text from each support ticket is sent to OpenAI's embedding models, which translate human language into high-dimensional numerical vectors.
* **Storage:** These vectors are stored locally in a highly efficient **Chroma** vector database. 
* **Purpose:** This mathematical representation allows the system to organize data by conceptual meaning. The AI can later search the database based on the intent of a query, rather than relying on brittle, exact-keyword matching.

### 3. The Retriever (Context Gathering)
* **Action:** When a user submits a question, the application vectorizes the query and performs a similarity search against the Chroma database.
* **Constraint:** It isolates and extracts the **top 3 most relevant** support tickets containing data related to the user's specific problem.

### 4. The Generator (Grounded Response Creation)
* **Action:** The user's prompt and the 3 retrieved context blocks are packaged together and sent to the LLM (`gpt-4o-mini`).
* **Guardrails:** The system operates under a strict system prompt instruction: 
  > *"You are a reliable customer support QA assistant. Use the following pieces of retrieved context to answer the user's question accurately. If the context does not contain the answer, explicitly state that you do not know."*

### 5. The Output (Structured Payload)
* **Action:** The application compiles its response into a structured Python dictionary.
* **Payload:** Rather than returning plain text, it outputs both the generated **`answer`** and the exact **`contexts`** (the raw text of the 3 tickets pulled from the database). Passing the raw context forward is critical, as it provides the exact transparency needed for our automated evaluation engine to audit the response.

## AI Quality Pipeline

🗂️ 1. Designing the Golden Test Set

To evaluate a RAG pipeline, you must first establish a baseline of truth. I designed a custom evaluation dataset (eval_baseline.csv) based on a dataset of real-world synthetic customer support tickets (support_tickets.csv).

My test set design strategy focused on three specific scenarios:
* Direct Extractions: Queries that map perfectly to a single ticket (e.g., "What happens when I open the settings tab?"). This tests the retriever's baseline capability.
* Synthesis Queries: Questions requiring the AI to combine context from the data.
* Adversarial / Out-of-Scope Queries: I deliberately included questions completely unrelated to customer support (e.g., "What is the company policy for employee weekend overtime pay?"). The ground_truth for these was mapped to "I do not know." This is a critical QA step to ensure the AI gracefully handles missing context without hallucinating.


⚖️ 2. Measuring Quality with Ragas

Once the test set was designed, I built an automated evaluation script (evaluate_metrics.py) that queries the live RAG application and scores the outputs using the Ragas framework. I focused on three primary metrics:

🎯 Context Precision (Evaluating the Retriever)

* What it measures: The signal-to-noise ratio of the retrieved documents.
* How it works: The evaluator looks at the user's question, the perfect ground_truth, and the raw contexts (the actual tickets pulled from the local Chroma vector database). It penalizes the system if the database retrieved irrelevant tickets or ranked the correct ticket too low.


🛡️ Faithfulness (Evaluating the Generator / Hallucination Detection)
* What it measures: The factual accuracy of the generated response.
* How it works: The evaluator cross-references the generated answer strictly against the retrieved contexts. If the LLM generates a response claiming "Refunds take 24 hours," but that timeline is not explicitly stated in the retrieved ticket, the Faithfulness score drops. This ensures the AI remains strictly grounded in approved company policies.


💬 Answer Relevancy (Evaluating the User Experience)
* What it measures: How directly the AI addressed the user's specific prompt.
* How it works: The evaluator analyzes the question and the answer to penalize evasive, incomplete, or overly verbose responses. A highly faithful answer is useless if it doesn't actually solve the customer's problem.

## 🚀 How to Run the Pipeline
1. Clone the repository and activate your virtual environment.
2. Install dependencies: pip install -r requirements.txt
3. Add your OpenAI API key to a .env file.
4. Run the evaluation engine: python evaluate_metrics.py
5. Review the generated evaluation_results.csv for a row-by-row breakdown of the RAG pipeline's performance.
