# Quick Start Guide - Agentic RAG System

Get your 2-agent RAG system up and running in 5 minutes!

## Step 1: Verify Setup

Run the test script to ensure everything is configured correctly:

```bash
python test_agentic_rag.py
```

Expected output:
```
============================================================
AGENTIC RAG SYSTEM - SETUP VERIFICATION
============================================================

Testing environment variables...
  ✓ All environment variables found

Testing Pinecone connection...
  ✓ Connected to index: traditional-rag-index
  ✓ Total vectors: 38735
  ✓ Namespace 'no-excel' found with XXXX vectors

Testing embedding model...
  ✓ Model loaded successfully
  ✓ Device: cuda (or cpu)
  ✓ Embedding dimension: 768

Testing Gemini API...
  ✓ Gemini API working
  ✓ Test response: Hello, World!...

Testing custom agent imports...
  ✓ All agent classes imported successfully

============================================================
TEST SUMMARY
============================================================
✓ PASS - Environment Variables
✓ PASS - Pinecone Connection
✓ PASS - Embedding Model
✓ PASS - Gemini API
✓ PASS - Agent Imports

Total: 5/5 tests passed

🎉 All tests passed! Your system is ready to use.
```

**If any test fails:**
- Check your `.env` file has all required variables
- Verify API keys are valid
- Ensure `no-excel` namespace exists in your Pinecone index

## Step 2: Open the Demo Notebook

```bash
jupyter notebook agentic_rag_demo.ipynb
```

Or if using VS Code, open the notebook in the editor.

## Step 3: Run the Notebook

Execute all cells in order (Cell → Run All). The notebook will:

1. **Initialize** agents and models
2. **Process** 7 sample questions
3. **Evaluate** results against expected answers
4. **Export** results to CSV
5. **Visualize** performance metrics

## Step 4: View Results

After running, check the `results/` directory:

```bash
ls -lh results/
```

You'll find:
- `agentic_rag_results_TIMESTAMP.csv` - Evaluation results
- `agentic_rag_logs_TIMESTAMP.json` - Detailed logs
- `agentic_rag_visualization_TIMESTAMP.png` - Charts

## Sample Output

### Console Output During Processing

```
================================================================================
# Processing 7 questions
================================================================================

[Orchestrator] Processing question: 'What is the Corporate Identity Number...'
================================================================================

--- Iteration 1/3 ---
[RetrieverAgent] Encoding query: 'What is the Corporate Identity Number...'
[RetrieverAgent] Querying Pinecone namespace='no-excel' with top_k=5
[RetrieverAgent] Retrieved 5 matches
[RetrieverAgent] Returning 5 new chunks

[EvaluatorAgent] Checking context sufficiency...
[EvaluatorAgent] Context is SUFFICIENT
[EvaluatorAgent] Reasoning: The context contains the CIN directly...

[EvaluatorAgent] Generating final answer...
[EvaluatorAgent] Generated answer (87 chars)

[Orchestrator] Answer generated successfully in 1 iteration(s)
```

### CSV Results Preview

| Question_Number | Question | Expected_Answer | Generated_Answer | Iterations | Success |
|-----------------|----------|-----------------|------------------|------------|---------|
| 1 | What is the CIN... | U85110KA1991... | The CIN is U85110KA1991... | 1 | True |
| 2 | What is the authorised... | 5,000,000.0 Rs. | The authorised capital is 5,000,000... | 1 | True |

## Understanding the Agent Interaction

### Example: Question Requiring Multiple Iterations

**Question:** "When was the AGM held?"

**Iteration 1:**
- Retriever gets 5 chunks
- Evaluator: "INSUFFICIENT - no date found"
- Refinement: "Try searching for 'Annual General Meeting date'"

**Iteration 2:**
- Retriever gets 5 more chunks (total: 10)
- Evaluator: "SUFFICIENT - found AGM date"
- Generates answer: "The AGM was held on 30/09/2024"

## Customization

### Adjust Iteration Settings

Edit the orchestrator initialization in the notebook:

```python
orchestrator = AgenticRAGOrchestrator(
    retriever=retriever,
    evaluator=evaluator,
    max_iterations=3,        # Increase for more attempts
    initial_top_k=5,         # Start with more chunks
    top_k_increment=5        # Larger increments per iteration
)
```

### Change Namespace

If you want to use a different Pinecone namespace:

```python
NAMESPACE = "your-namespace-name"
```

### Use Different Questions

Replace `sample_questions` list in cell 6:

```python
sample_questions = [
    "Your custom question 1?",
    "Your custom question 2?",
    # Add more...
]
```

## Troubleshooting

### "No chunks retrieved"

**Problem:** Pinecone returns no results

**Solutions:**
1. Verify `no-excel` namespace has data:
   ```python
   stats = index.describe_index_stats()
   print(stats['namespaces'])
   ```
2. Try a broader query
3. Check embedding model matches index (768 dimensions)

### "Max iterations reached"

**Problem:** Agent can't find sufficient context in 3 iterations

**Solutions:**
1. Increase `max_iterations` to 5
2. Increase `initial_top_k` to 10
3. Review logs to see what was retrieved
4. Question may be too specific or data missing

### Gemini API Rate Limits

**Problem:** `429 Too Many Requests` error

**Solutions:**
1. Add delays between questions:
   ```python
   import time
   for q in questions:
       result = orchestrator.process_question(q)
       time.sleep(2)  # 2 second delay
   ```
2. Reduce number of questions per batch
3. Upgrade Gemini API quota

## Next Steps

1. **Experiment** with different questions
2. **Analyze** the logs to understand agent decisions
3. **Tune** parameters based on your use case
4. **Extend** the system with custom logic

## Code Example: Single Question

Want to test just one question? Use this code:

```python
from agentic_rag import RetrieverAgent, EvaluatorAgent, AgenticRAGOrchestrator
# ... (initialization code from notebook) ...

# Process single question
result = orchestrator.process_question(
    "What is the Corporate Identity Number (CIN) of the company?"
)

print(f"Answer: {result['answer']}")
print(f"Iterations: {result['iterations']}")
print(f"Success: {result['success']}")
```

## Performance Tips

1. **Use GPU** if available (automatic with PyTorch/CUDA)
2. **Batch questions** efficiently with `process_questions_batch()`
3. **Cache embeddings** for repeated queries (future enhancement)
4. **Monitor** API usage to avoid rate limits

## Support

For issues or questions:
1. Check the full [README_AGENTIC_RAG.md](README_AGENTIC_RAG.md)
2. Review error logs in `results/` directory
3. Run `test_agentic_rag.py` to verify setup

---

**Ready to start?** Run `python test_agentic_rag.py` now!
