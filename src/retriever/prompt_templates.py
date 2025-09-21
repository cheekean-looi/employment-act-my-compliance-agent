#!/usr/bin/env python3
# python src/retriever/prompt_templates.py
"""
Prompt templates for Employment Act Malaysia compliance agent.
Enforces citation requirements and structured JSON output.
"""

from typing import List, Dict, Any
import json
from dataclasses import dataclass


@dataclass
class Citation:
    """Citation with section ID and snippet."""
    section_id: str
    snippet: str


@dataclass
class Response:
    """Structured response with answer, citations, confidence, and escalation flag."""
    answer: str
    citations: List[Citation]
    confidence: float
    should_escalate: bool
    safety_flags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "citations": [{"section_id": c.section_id, "snippet": c.snippet} for c in self.citations],
            "confidence": self.confidence,
            "should_escalate": self.should_escalate,
            "safety_flags": self.safety_flags or []
        }


class PromptTemplates:
    """Employment Act Malaysia prompt templates with citation enforcement."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt defining the assistant's role and behavior."""
        return """You are a specialized Employment Act Malaysia compliance assistant. Your role is to provide accurate, legally-grounded answers based ONLY on the provided context from official Employment Act documents.

CRITICAL REQUIREMENTS:
1. Use ONLY information from the provided context - never use external knowledge
2. Every claim must be supported with specific section citations
3. If information is insufficient, say "I'm not certain; here's the relevant section and how to proceed"
4. Output must be valid JSON with the exact schema specified
5. Confidence score must reflect certainty in the answer based on context quality
6. Flag for escalation if question is complex or context is insufficient

CITATION FORMAT:
- Use exact section IDs from the context (e.g., "EA-2022-15(1)", "Section 25A")
- Include relevant text snippets that support your answer
- Every factual claim needs a citation

RESPONSE SCHEMA:
{
  "answer": "Your detailed answer here",
  "citations": [{"section_id": "EA-2022-15(1)", "snippet": "relevant text..."}],
  "confidence": 0.85,
  "should_escalate": false,
  "safety_flags": []
}

Remember: You are a legal compliance tool. Accuracy and proper citation are paramount."""

    @staticmethod
    def get_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate RAG prompt with query and retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks with metadata
            
        Returns:
            Formatted prompt string
        """
        # Format context
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            section_id = chunk.get('section_id', 'N/A')
            text = chunk['text']
            context_text += f"[Context {i}] Section: {section_id}\n{text}\n\n"
        
        if not context_text.strip():
            context_text = "[No relevant context found]"
        
        prompt = f"""CONTEXT:
{context_text}

QUESTION: {query}

Based ONLY on the provided context, provide a comprehensive answer following the JSON response schema. If the context doesn't contain sufficient information to answer confidently, acknowledge this limitation and provide guidance on how to proceed.

Response:"""
        
        return prompt
    
    @staticmethod
    def get_refusal_prompt(query: str) -> str:
        """Generate refusal prompt for out-of-scope questions.
        
        Args:
            query: User's question
            
        Returns:
            Formatted refusal response
        """
        response = Response(
            answer="I can only provide guidance on Employment Act Malaysia matters. This question appears to be outside my scope of expertise. Please consult with a legal professional or relevant authority for questions beyond employment law.",
            citations=[],
            confidence=1.0,
            should_escalate=True,
            safety_flags=["out_of_scope"]
        )
        
        return json.dumps(response.to_dict(), indent=2)
    
    @staticmethod
    def get_insufficient_context_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate response for when context is insufficient.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks (may be low quality)
            
        Returns:
            Formatted response acknowledging limitations
        """
        # Find the most relevant section if any
        best_section = None
        if context_chunks:
            best_chunk = context_chunks[0]  # Assuming sorted by relevance
            best_section = best_chunk.get('section_id', 'N/A')
        
        answer = f"I'm not certain about this specific question based on the available context. "
        
        if best_section and best_section != 'N/A':
            answer += f"The most relevant section I found is {best_section}. "
        
        answer += "For accurate guidance on this matter, I recommend:\n\n"
        answer += "1. Consulting the complete Employment Act Malaysia 2022\n"
        answer += "2. Speaking with a qualified employment law practitioner\n"
        answer += "3. Contacting the Department of Labour Malaysia for official clarification\n\n"
        answer += "This ensures you receive legally sound advice for your specific situation."
        
        citations = []
        if context_chunks and best_section and best_section != 'N/A':
            citations = [Citation(
                section_id=best_section,
                snippet=context_chunks[0]['text'][:200] + "..."
            )]
        
        response = Response(
            answer=answer,
            citations=citations,
            confidence=0.3,
            should_escalate=True,
            safety_flags=["insufficient_context"]
        )
        
        return json.dumps(response.to_dict(), indent=2)
    
    @staticmethod
    def validate_response_format(response_text: str) -> bool:
        """Validate that response follows the required JSON schema.
        
        Args:
            response_text: Generated response text
            
        Returns:
            True if valid, False otherwise
        """
        try:
            data = json.loads(response_text)
            
            # Check required fields
            required_fields = ["answer", "citations", "confidence", "should_escalate"]
            for field in required_fields:
                if field not in data:
                    return False
            
            # Check types
            if not isinstance(data["answer"], str):
                return False
            if not isinstance(data["citations"], list):
                return False
            if not isinstance(data["confidence"], (int, float)):
                return False
            if not isinstance(data["should_escalate"], bool):
                return False
            
            # Check citation format
            for citation in data["citations"]:
                if not isinstance(citation, dict):
                    return False
                if "section_id" not in citation or "snippet" not in citation:
                    return False
            
            # Check confidence range
            if not (0.0 <= data["confidence"] <= 1.0):
                return False
            
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    
    @staticmethod
    def extract_response_data(response_text: str) -> Dict[str, Any]:
        """Extract and validate response data from generated text.
        
        Args:
            response_text: Generated response text
            
        Returns:
            Parsed response dictionary
            
        Raises:
            ValueError: If response format is invalid
        """
        # Try to find JSON in the response
        response_text = response_text.strip()
        
        # If response doesn't start with {, try to find JSON block
        if not response_text.startswith('{'):
            # Look for JSON block
            start_idx = response_text.find('{')
            if start_idx != -1:
                response_text = response_text[start_idx:]
        
        try:
            data = json.loads(response_text)
            
            if not PromptTemplates.validate_response_format(response_text):
                raise ValueError("Response does not match required schema")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")


def main():
    """Test prompt templates."""
    # Test system prompt
    print("SYSTEM PROMPT:")
    print(PromptTemplates.get_system_prompt())
    print("\n" + "="*60 + "\n")
    
    # Test RAG prompt
    sample_chunks = [
        {
            "section_id": "EA-2022-15(1)",
            "text": "An employee shall be entitled to paid annual leave of not less than 8 days for every 12 months of continuous service with the same employer."
        },
        {
            "section_id": "EA-2022-15(2)", 
            "text": "The annual leave shall be in addition to the rest days and public holidays."
        }
    ]
    
    query = "How many days of annual leave am I entitled to?"
    rag_prompt = PromptTemplates.get_rag_prompt(query, sample_chunks)
    print("RAG PROMPT:")
    print(rag_prompt)
    print("\n" + "="*60 + "\n")
    
    # Test response validation
    valid_response = {
        "answer": "You are entitled to at least 8 days of paid annual leave for every 12 months of continuous service.",
        "citations": [{"section_id": "EA-2022-15(1)", "snippet": "An employee shall be entitled to paid annual leave of not less than 8 days..."}],
        "confidence": 0.95,
        "should_escalate": False,
        "safety_flags": []
    }
    
    response_json = json.dumps(valid_response, indent=2)
    print("SAMPLE VALID RESPONSE:")
    print(response_json)
    print(f"Validation result: {PromptTemplates.validate_response_format(response_json)}")


if __name__ == "__main__":
    main()