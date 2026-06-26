**Automated RAG Quality Evaluation Pipeline**

🎯 Project Objective

The goal of this project is to build a highly reliable Retrieval-Augmented Generation (RAG) system for customer support and, crucially, to implement an automated quality assurance pipeline using the Ragas framework.
Rather than relying on manual spot-checking, this project demonstrates how to use LLM-as-a-judge methodologies to quantitatively measure hallucinations, retrieval accuracy, and response usefulness in production AI systems.

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

🚀 How to Run the Pipeline
1. Clone the repository and activate your virtual environment.
2. Install dependencies: pip install -r requirements.txt
3. Add your OpenAI API key to a .env file.
4. Run the evaluation engine: python evaluate_metrics.py
5. Review the generated evaluation_results.csv for a row-by-row breakdown of the RAG pipeline's performance.
