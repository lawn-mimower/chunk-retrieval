"""
Quick test script for Agentic RAG System

Run this to verify your setup before using the full notebook.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch

def test_environment_variables():
    """Test if all required environment variables are set."""
    print("Testing environment variables...")
    load_dotenv()

    required_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX_NAME', 'GEMINI_API_KEY']
    missing = []

    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"  ✗ Missing environment variables: {', '.join(missing)}")
        return False
    else:
        print("  ✓ All environment variables found")
        return True

def test_pinecone_connection():
    """Test Pinecone connection and namespace."""
    print("\nTesting Pinecone connection...")
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
        stats = index.describe_index_stats()

        print(f"  ✓ Connected to index: {os.getenv('PINECONE_INDEX_NAME')}")
        print(f"  ✓ Total vectors: {stats['total_vector_count']}")

        # Check for no-excel namespace
        if 'no-excel' in stats.get('namespaces', {}):
            count = stats['namespaces']['no-excel']['vector_count']
            print(f"  ✓ Namespace 'no-excel' found with {count} vectors")
            return True
        else:
            print(f"  ⚠ Warning: 'no-excel' namespace not found")
            print(f"  Available namespaces: {list(stats.get('namespaces', {}).keys())}")
            return False

    except Exception as e:
        print(f"  ✗ Pinecone error: {e}")
        return False

def test_embedding_model():
    """Test embedding model loading."""
    print("\nTesting embedding model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer("all-mpnet-base-v2", device=device)

        # Test encoding
        test_text = "This is a test sentence"
        embedding = model.encode(test_text, normalize_embeddings=True)

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Device: {device}")
        print(f"  ✓ Embedding dimension: {len(embedding)}")
        return True

    except Exception as e:
        print(f"  ✗ Embedding model error: {e}")
        return False

def test_gemini_api():
    """Test Gemini API connection."""
    print("\nTesting Gemini API...")
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Test generation
        response = model.generate_content("Say 'Hello, World!' in exactly those words.")

        print(f"  ✓ Gemini API working")
        print(f"  ✓ Test response: {response.text[:50]}...")
        return True

    except Exception as e:
        print(f"  ✗ Gemini API error: {e}")
        return False

def test_agent_imports():
    """Test if custom agent modules can be imported."""
    print("\nTesting custom agent imports...")
    try:
        from agentic_rag import RetrieverAgent, EvaluatorAgent, AgenticRAGOrchestrator
        print("  ✓ All agent classes imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("AGENTIC RAG SYSTEM - SETUP VERIFICATION")
    print("="*60 + "\n")

    tests = [
        ("Environment Variables", test_environment_variables),
        ("Pinecone Connection", test_pinecone_connection),
        ("Embedding Model", test_embedding_model),
        ("Gemini API", test_gemini_api),
        ("Agent Imports", test_agent_imports),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("Run 'agentic_rag_demo.ipynb' to start processing questions.")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")

    print("="*60)

if __name__ == "__main__":
    run_all_tests()
