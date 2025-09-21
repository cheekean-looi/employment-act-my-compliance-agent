"""
Employment Act Malaysia retrieval and RAG pipeline.

This module provides hybrid retrieval (BM25 + dense + cross-encoder re-ranking),
prompt templates with citation enforcement, guardrails, and complete RAG pipeline.
"""

from .hybrid_retriever import HybridRetriever
from .prompt_templates import PromptTemplates, Response, Citation
from .guardrails import EmploymentActGuardrails, SafetyFlag, GuardrailsResult
from .rag_pipeline import EmploymentActRAG, RAGConfig

__all__ = [
    'HybridRetriever',
    'PromptTemplates',
    'Response',
    'Citation',
    'EmploymentActGuardrails',
    'SafetyFlag',
    'GuardrailsResult',
    'EmploymentActRAG',
    'RAGConfig'
]