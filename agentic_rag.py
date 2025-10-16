"""
Agentic RAG System with Retriever and Evaluator Agents

This module implements a 2-agent system where:
1. RetrieverAgent: Retrieves chunks from Pinecone based on queries
2. EvaluatorAgent: Evaluates sufficiency and generates answers using Gemini
3. AgenticRAGOrchestrator: Coordinates the agents in an iterative loop
"""

import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import google.generativeai as genai


class RetrieverAgent:
    """
    Agent responsible for retrieving relevant chunks from Pinecone vector store.
    Uses the 'no-excel' namespace and tracks retrieved chunks to avoid duplicates.
    """

    def __init__(self, index, embedding_model, namespace: str = "no-excel"):
        """
        Initialize the Retriever Agent.

        Args:
            index: Pinecone index object
            embedding_model: SentenceTransformer model for embeddings
            namespace: Pinecone namespace to query (default: "no-excel")
        """
        self.index = index
        self.model = embedding_model
        self.namespace = namespace
        self.retrieved_chunk_ids = set()  # Track retrieved chunks

    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        return_metadata: bool = True
    ) -> Dict:
        """
        Retrieve relevant chunks from Pinecone.

        Args:
            query: Search query string
            top_k: Number of chunks to retrieve
            return_metadata: Whether to include metadata

        Returns:
            Dictionary containing:
                - chunks: List of retrieved text chunks
                - chunk_ids: List of chunk IDs
                - scores: List of similarity scores
                - metadata: List of metadata dicts (if return_metadata=True)
        """
        print(f"[RetrieverAgent] Encoding query: '{query[:50]}...'")
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()

        print(f"[RetrieverAgent] Querying Pinecone namespace='{self.namespace}' with top_k={top_k}")
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True
        )

        chunks = []
        chunk_ids = []
        scores = []
        metadata_list = []

        if results['matches']:
            print(f"[RetrieverAgent] Retrieved {len(results['matches'])} matches")
            for match in results['matches']:
                chunk_id = match['id']

                # Skip if already retrieved
                if chunk_id in self.retrieved_chunk_ids:
                    continue

                self.retrieved_chunk_ids.add(chunk_id)
                chunk_ids.append(chunk_id)
                scores.append(match['score'])

                if 'metadata' in match and 'text' in match['metadata']:
                    chunks.append(match['metadata']['text'])
                    if return_metadata:
                        metadata_list.append(match['metadata'])
        else:
            print("[RetrieverAgent] No matches found")

        print(f"[RetrieverAgent] Returning {len(chunks)} new chunks (excluded {len(results['matches']) - len(chunks)} duplicates)")

        return {
            'chunks': chunks,
            'chunk_ids': chunk_ids,
            'scores': scores,
            'metadata': metadata_list if return_metadata else None,
            'total_retrieved_so_far': len(self.retrieved_chunk_ids)
        }

    def refine_query(self, original_query: str, feedback: str) -> str:
        """
        Refine the query based on feedback from the Evaluator.

        Args:
            original_query: Original search query
            feedback: Feedback/suggestions from Evaluator

        Returns:
            Refined query string
        """
        # Simple refinement: append feedback to original query
        refined = f"{original_query} {feedback}".strip()
        print(f"[RetrieverAgent] Refined query: '{refined[:100]}...'")
        return refined

    def reset_tracking(self):
        """Reset the tracking of retrieved chunks for a new question."""
        self.retrieved_chunk_ids.clear()
        print("[RetrieverAgent] Reset chunk tracking")


class EvaluatorAgent:
    """
    Agent responsible for evaluating retrieved context and generating answers.
    Uses Gemini API to assess sufficiency and generate responses.
    """

    def __init__(self, gemini_model):
        """
        Initialize the Evaluator Agent.

        Args:
            gemini_model: Initialized Gemini GenerativeModel object
        """
        self.model = gemini_model

    def check_sufficiency(self, query: str, context: str) -> Tuple[bool, str, Optional[str]]:
        """
        Check if the retrieved context is sufficient to answer the query.

        Args:
            query: User's question
            context: Retrieved context chunks

        Returns:
            Tuple of (is_sufficient, reasoning, refinement_suggestion)
        """
        if not context or not context.strip():
            return False, "No context retrieved", "Try broader search terms or different keywords"

        sufficiency_prompt = f"""You are an expert evaluator assessing whether retrieved context is sufficient to answer a query.

QUERY: {query}

RETRIEVED CONTEXT:
---
{context}
---

YOUR TASK:
1. Carefully analyze if the context contains enough information to fully answer the query
2. Respond with a JSON object in this exact format:
{{
    "sufficient": true/false,
    "reasoning": "Your detailed explanation",
    "missing_info": "What information is missing (if insufficient)",
    "refinement_suggestion": "Suggested keywords or query modifications (if insufficient)"
}}

EVALUATION CRITERIA:
- SUFFICIENT: Context contains direct answer or all necessary information
- INSUFFICIENT: Answer is unclear, incomplete, or information is missing

Respond ONLY with the JSON object, no other text."""

        try:
            print("[EvaluatorAgent] Checking context sufficiency...")
            response = self.model.generate_content(sufficiency_prompt)
            response_text = response.text.strip()

            # Extract JSON from response (handle potential markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            is_sufficient = result.get('sufficient', False)
            reasoning = result.get('reasoning', '')
            refinement = result.get('refinement_suggestion', '')

            status = "SUFFICIENT" if is_sufficient else "INSUFFICIENT"
            print(f"[EvaluatorAgent] Context is {status}")
            print(f"[EvaluatorAgent] Reasoning: {reasoning[:100]}...")

            return is_sufficient, reasoning, refinement if not is_sufficient else None

        except json.JSONDecodeError as e:
            print(f"[EvaluatorAgent] JSON parsing error: {e}")
            print(f"[EvaluatorAgent] Raw response: {response_text[:200]}")
            # Fallback: assume insufficient and suggest generic refinement
            return False, "Error parsing evaluation response", "Try more specific keywords"
        except Exception as e:
            print(f"[EvaluatorAgent] Error during sufficiency check: {e}")
            return False, f"Error: {str(e)}", "Try alternative search terms"

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate final answer using the retrieved context.

        Args:
            query: User's question
            context: Retrieved context chunks

        Returns:
            Generated answer string
        """
        if not context:
            return "I could not find any relevant information in the documents to answer your question."

        answer_prompt = f"""You are a helpful and accurate question-answering assistant.
Your task is to answer the user's query based ONLY on the provided context.

IMPORTANT RULES:
1. Use ONLY information from the context provided
2. Do not use external knowledge
3. Be concise and direct
4. If the context doesn't contain the answer, state that clearly
5. Cite source files if mentioned in the context

CONTEXT:
---
{context}
---

QUERY: {query}

ANSWER:"""

        try:
            print("[EvaluatorAgent] Generating final answer...")
            response = self.model.generate_content(answer_prompt)
            answer = response.text.strip()
            print(f"[EvaluatorAgent] Generated answer ({len(answer)} chars)")
            return answer

        except Exception as e:
            print(f"[EvaluatorAgent] Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"


class AgenticRAGOrchestrator:
    """
    Orchestrates the interaction between Retriever and Evaluator agents.
    Implements an iterative loop to improve retrieval until sufficient context is found.
    """

    def __init__(
        self,
        retriever: RetrieverAgent,
        evaluator: EvaluatorAgent,
        max_iterations: int = 3,
        initial_top_k: int = 5,
        top_k_increment: int = 5
    ):
        """
        Initialize the orchestrator.

        Args:
            retriever: RetrieverAgent instance
            evaluator: EvaluatorAgent instance
            max_iterations: Maximum number of retrieval iterations
            initial_top_k: Initial number of chunks to retrieve
            top_k_increment: How much to increase top_k each iteration
        """
        self.retriever = retriever
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.initial_top_k = initial_top_k
        self.top_k_increment = top_k_increment
        self.interaction_logs = []

    def process_question(self, query: str) -> Dict:
        """
        Process a single question through the agentic RAG pipeline.

        Args:
            query: User's question

        Returns:
            Dictionary containing:
                - question: Original query
                - answer: Generated answer
                - iterations: Number of iterations performed
                - success: Whether answer was generated
                - logs: Detailed interaction logs
        """
        print(f"\n{'='*80}")
        print(f"[Orchestrator] Processing question: '{query}'")
        print(f"{'='*80}\n")

        # Reset retriever tracking for new question
        self.retriever.reset_tracking()

        iteration_logs = []
        current_query = query
        all_context_chunks = []
        top_k = self.initial_top_k

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            iteration_log = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'query': current_query,
                'top_k': top_k
            }

            # Step 1: Retrieve chunks
            retrieval_result = self.retriever.retrieve_chunks(
                query=current_query,
                top_k=top_k
            )

            new_chunks = retrieval_result['chunks']
            all_context_chunks.extend(new_chunks)

            iteration_log['chunks_retrieved'] = len(new_chunks)
            iteration_log['total_chunks'] = len(all_context_chunks)

            if not all_context_chunks:
                print("[Orchestrator] No chunks retrieved. Cannot proceed.")
                iteration_log['outcome'] = 'no_chunks'
                iteration_logs.append(iteration_log)
                break

            # Combine all context
            combined_context = "\n\n---\n\n".join(all_context_chunks)

            # Step 2: Check sufficiency
            is_sufficient, reasoning, refinement = self.evaluator.check_sufficiency(
                query=query,  # Use original query for evaluation
                context=combined_context
            )

            iteration_log['sufficient'] = is_sufficient
            iteration_log['reasoning'] = reasoning

            if is_sufficient:
                # Step 3: Generate answer
                answer = self.evaluator.generate_answer(
                    query=query,
                    context=combined_context
                )

                iteration_log['answer'] = answer
                iteration_log['outcome'] = 'success'
                iteration_logs.append(iteration_log)

                print(f"\n[Orchestrator] Answer generated successfully in {iteration} iteration(s)")

                return {
                    'question': query,
                    'answer': answer,
                    'iterations': iteration,
                    'total_chunks_used': len(all_context_chunks),
                    'success': True,
                    'logs': iteration_logs
                }

            else:
                # Not sufficient - prepare for next iteration
                iteration_log['refinement_suggestion'] = refinement
                iteration_log['outcome'] = 'insufficient'
                iteration_logs.append(iteration_log)

                if iteration < self.max_iterations:
                    print(f"[Orchestrator] Context insufficient. Preparing iteration {iteration + 1}...")

                    # Refine query for next iteration
                    if refinement:
                        current_query = self.retriever.refine_query(query, refinement)

                    # Increase top_k for next iteration
                    top_k += self.top_k_increment
                else:
                    print("[Orchestrator] Max iterations reached. Generating best-effort answer...")

                    # Generate answer with what we have
                    answer = self.evaluator.generate_answer(
                        query=query,
                        context=combined_context
                    )

                    return {
                        'question': query,
                        'answer': answer,
                        'iterations': iteration,
                        'total_chunks_used': len(all_context_chunks),
                        'success': False,
                        'logs': iteration_logs,
                        'note': 'Max iterations reached - answer may be incomplete'
                    }

        # Fallback if loop exits without answer
        return {
            'question': query,
            'answer': "Unable to generate answer due to insufficient context.",
            'iterations': len(iteration_logs),
            'total_chunks_used': len(all_context_chunks),
            'success': False,
            'logs': iteration_logs
        }

    def process_questions_batch(self, questions: List[str]) -> List[Dict]:
        """
        Process multiple questions sequentially.

        Args:
            questions: List of question strings

        Returns:
            List of result dictionaries
        """
        results = []

        print(f"\n{'#'*80}")
        print(f"# Processing {len(questions)} questions")
        print(f"{'#'*80}\n")

        for idx, question in enumerate(questions, 1):
            print(f"\n[Orchestrator] Question {idx}/{len(questions)}")
            result = self.process_question(question)
            results.append(result)

        print(f"\n{'#'*80}")
        print(f"# Completed processing {len(questions)} questions")
        print(f"# Success rate: {sum(r['success'] for r in results)}/{len(questions)}")
        print(f"{'#'*80}\n")

        return results
