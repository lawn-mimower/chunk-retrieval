# Agentic RAG System

A 2-agent Retrieval-Augmented Generation (RAG) system using Pinecone and Gemini LLM API.

## Overview

This system implements an iterative RAG pipeline with two cooperating agents:

1. **RetrieverAgent**: Retrieves relevant chunks from Pinecone vector database (namespace: `no-excel`)
2. **EvaluatorAgent**: Evaluates context sufficiency and generates answers using Gemini API

The agents work together in an iterative loop until sufficient context is found or max iterations are reached.

## Architecture

```
User Question
     ↓
┌────────────────────────────────────────┐
│   AgenticRAGOrchestrator              │
│                                        │
│  ┌──────────────────────────────┐    │
│  │ Iteration Loop (max 3)       │    │
│  │                              │    │
│  │  1. RetrieverAgent           │    │
│  │     → Query Pinecone         │    │
│  │     → Get chunks             │    │
│  │                              │    │
│  │  2. EvaluatorAgent           │    │
│  │     → Check sufficiency      │    │
│  │     → If insufficient:       │    │
│  │       • Suggest refinement   │    │
│  │       • Loop back            │    │
│  │     → If sufficient:         │    │
│  │       • Generate answer      │    │
│  │       • Return result        │    │
│  └──────────────────────────────┘    │
└────────────────────────────────────────┘
     ↓
Final Answer
```

## Files

- **`agentic_rag.py`**: Core implementation
  - `RetrieverAgent`: Handles Pinecone queries
  - `EvaluatorAgent`: Handles Gemini API calls
  - `AgenticRAGOrchestrator`: Coordinates agents

- **`agentic_rag_demo.ipynb`**: Interactive demo notebook
  - Processes sample questions
  - Compares with expected answers
  - Exports results to CSV
  - Generates visualizations

- **`results/`**: Output directory
  - CSV files with evaluation results
  - JSON logs with detailed iteration history
  - Visualization PNG files

## Setup

### Prerequisites

```bash
pip install pandas python-dotenv torch sentence-transformers pinecone-client google-generativeai matplotlib
```

### Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_INDEX_NAME="your_index_name"
GEMINI_API_KEY="your_gemini_api_key"
```

## Usage

### Option 1: Using the Jupyter Notebook (Recommended)

1. Open `agentic_rag_demo.ipynb`
2. Run all cells sequentially
3. View results in the notebook and in `results/` directory

### Option 2: Using Python Script

```python
from agentic_rag import RetrieverAgent, EvaluatorAgent, AgenticRAGOrchestrator
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

# Initialize components
pc = Pinecone(api_key="your_api_key")
index = pc.Index("your_index_name")
model = SentenceTransformer("all-mpnet-base-v2")

genai.configure(api_key="your_gemini_key")
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Create agents
retriever = RetrieverAgent(index, model, namespace="no-excel")
evaluator = EvaluatorAgent(gemini_model)
orchestrator = AgenticRAGOrchestrator(retriever, evaluator)

# Process questions
questions = ["Your question here"]
results = orchestrator.process_questions_batch(questions)

# Access results
for result in results:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {result['iterations']}")
```

## How It Works

### Iteration Flow

1. **Initial Retrieval** (Iteration 1)
   - RetrieverAgent queries Pinecone with `top_k=5`
   - Retrieves most relevant chunks from `no-excel` namespace

2. **Sufficiency Check**
   - EvaluatorAgent sends context + query to Gemini
   - Gemini responds with:
     - `sufficient`: true/false
     - `reasoning`: Explanation
     - `refinement_suggestion`: How to improve query (if insufficient)

3. **Decision Point**
   - **If Sufficient**: EvaluatorAgent generates final answer → Done!
   - **If Insufficient**:
     - Refine query based on suggestions
     - Increase `top_k` by 5
     - Retrieve more chunks (iteration 2)

4. **Repeat** up to 3 iterations

5. **Fallback**
   - If max iterations reached without sufficiency
   - Generate best-effort answer with available context

### Key Features

- **Duplicate Avoidance**: Tracks retrieved chunk IDs across iterations
- **Progressive Retrieval**: Increases `top_k` each iteration (5 → 10 → 15)
- **Query Refinement**: Appends Evaluator's suggestions to query
- **Detailed Logging**: Tracks all interactions for debugging
- **Namespace Support**: Uses `no-excel` namespace in Pinecone

## Configuration

### Orchestrator Parameters

```python
orchestrator = AgenticRAGOrchestrator(
    retriever=retriever,
    evaluator=evaluator,
    max_iterations=3,        # Maximum retrieval attempts
    initial_top_k=5,         # Initial chunks to retrieve
    top_k_increment=5        # Increase per iteration
)
```

### Gemini Configuration

```python
generation_config = {
    "temperature": 0.2,       # Lower = more deterministic
    "top_p": 1,
    "top_k": 5,
    "max_output_tokens": 2048
}
```

## Sample Questions

The demo includes 7 sample questions about company filing information:

1. Corporate Identity Number (CIN)
2. Authorised capital
3. Annual General Meeting date
4. Registered office location
5. Secretarial Audit applicability
6. Financial year coverage
7. Certifier professional status

## Output Files

### CSV Results (`agentic_rag_results_TIMESTAMP.csv`)

| Column | Description |
|--------|-------------|
| Question_Number | Sequential question ID |
| Question | Original question text |
| Expected_Answer | Ground truth answer |
| Generated_Answer | AI-generated answer |
| Iterations | Number of retrieval iterations |
| Total_Chunks_Used | Total chunks retrieved |
| Success | Whether context was deemed sufficient |
| Note | Additional notes (e.g., "max iterations") |

### JSON Logs (`agentic_rag_logs_TIMESTAMP.json`)

Detailed iteration-by-iteration logs including:
- Queries at each iteration
- Retrieved chunk counts
- Sufficiency check results
- Refinement suggestions
- Final answers

## Troubleshooting

### "No chunks retrieved"
- Check if `no-excel` namespace exists in your Pinecone index
- Verify embedding model matches index dimensions (768)
- Check API keys and network connectivity

### "Max iterations reached"
- Questions may be too specific or data missing
- Try adjusting `max_iterations` or `top_k_increment`
- Review logs to see what context was retrieved

### Gemini API Errors
- Check API key validity
- Verify rate limits
- Review error messages in logs

## Performance Metrics

From demo results:
- **Success Rate**: Percentage of questions with sufficient context
- **Average Iterations**: Typically 1-2 for well-matched questions
- **Average Chunks Used**: 5-15 depending on complexity

## Future Enhancements

- [ ] Add semantic similarity scoring between expected/generated answers
- [ ] Implement re-ranking of retrieved chunks
- [ ] Add multi-query expansion strategies
- [ ] Support for multiple namespaces
- [ ] Implement caching for repeated queries
- [ ] Add conversation history for follow-up questions

## License

This project is for educational/research purposes.

## Contact

For issues or questions, please refer to the project repository.
