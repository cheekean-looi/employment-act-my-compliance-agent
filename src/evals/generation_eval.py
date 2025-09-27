#!/usr/bin/env python3
"""
Generation Evaluation System for Employment Act Malaysia Compliance Agent
LLM-as-judge scoring for groundedness and refusal evaluation.
"""

import json
import asyncio
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime
import statistics
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import shared validators
from evals.validation_utils import CitationValidator, NumericValidator

@dataclass
class EvaluationResult:
    """Result from LLM evaluation."""
    score: float
    reasoning: str
    category: str
    binary_label: bool
    confidence: float

@dataclass
class GroundednessResult:
    """Result from groundedness evaluation."""
    is_grounded: bool
    groundedness_score: float
    hallucination_detected: bool
    evidence_citations: List[str]
    reasoning: str

class LLMJudge:
    """
    LLM-as-judge for evaluating response quality, groundedness, and appropriateness.
    Uses smaller model (Qwen or base model) as the judge with shared validators.
    """
    
    def __init__(self, judge_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize LLM judge with specified model and shared validators."""
        self.judge_model_name = judge_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🏛️ Initializing LLM Judge: {judge_model}")
        
        # Load judge model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            judge_model,
            torch_dtype=torch.float32,
        )
        
        # Initialize shared validators
        self.citation_validator = CitationValidator()
        self.numeric_validator = NumericValidator()
        
        print(f"✅ Judge model loaded on {self.device}")
        print(f"📚 Citation validator initialized with {len(self.citation_validator.known_sections)} sections")
        print(f"🔢 Numeric validator initialized with {len(self.numeric_validator.bounds)} bound categories")
    
    def evaluate_groundedness(self, 
                            question: str, 
                            response: str, 
                            reference_context: str = "") -> GroundednessResult:
        """
        Evaluate if response is grounded in Employment Act provisions.
        Detects hallucinations and unsupported claims.
        """
        
        prompt = self._create_groundedness_prompt(question, response, reference_context)
        judge_response = self._query_judge(prompt)
        
        # Parse judge response
        is_grounded, score, reasoning = self._parse_groundedness_response(judge_response)
        
        # Use shared validators for citation and numeric validation
        citation_result = self.citation_validator.validate_citations(response)
        numeric_result = self.numeric_validator.validate_numeric_claims(response)
        
        # Detect hallucinations based on validation results
        hallucination_detected = (
            len(citation_result.invalid_citations) > 0 or 
            len(numeric_result.out_of_bounds) > 0 or
            self._detect_content_hallucinations(response)
        )
        
        # Adjust groundedness based on validation results
        if hallucination_detected and is_grounded:
            is_grounded = False
            score = min(score, 0.3)  # Cap score for responses with hallucinations
            reasoning += f" [Validation found: {len(citation_result.invalid_citations)} invalid citations, {len(numeric_result.out_of_bounds)} out-of-bounds claims]"
        
        return GroundednessResult(
            is_grounded=is_grounded,
            groundedness_score=score,
            hallucination_detected=hallucination_detected,
            evidence_citations=citation_result.valid_citations,
            reasoning=reasoning
        )
    
    def evaluate_refusal_appropriateness(self, 
                                       question: str, 
                                       response: str) -> EvaluationResult:
        """
        Evaluate if refusal is appropriate for the given question.
        Determines if model should have refused or answered.
        """
        
        prompt = self._create_refusal_prompt(question, response)
        judge_response = self._query_judge(prompt)
        
        # Parse response for refusal appropriateness
        is_appropriate, score, reasoning = self._parse_refusal_response(judge_response)
        
        return EvaluationResult(
            score=score,
            reasoning=reasoning,
            category="refusal_appropriateness",
            binary_label=is_appropriate,
            confidence=0.8  # Conservative confidence
        )
    
    def _create_groundedness_prompt(self, question: str, response: str, context: str) -> str:
        """Create prompt for groundedness evaluation."""
        
        prompt_template = """You are evaluating whether a response about Malaysia's Employment Act is grounded in legal facts.

Question: {question}

Response to evaluate: {response}

Reference context (if available): {context}

Evaluate the response on these criteria:
1. Factual accuracy about Employment Act provisions
2. Presence of proper legal citations (Section EA-X)
3. Absence of hallucinated information
4. Appropriate disclaimers and limitations

Rate the groundedness from 0.0 (completely ungrounded/hallucinated) to 1.0 (fully grounded).

Respond in this format:
GROUNDED: [YES/NO]
SCORE: [0.0-1.0]
REASONING: [Your detailed explanation]"""

        return prompt_template.format(
            question=question,
            response=response,
            context=context if context else "No specific context provided"
        )
    
    def _create_refusal_prompt(self, question: str, response: str) -> str:
        """Create prompt for refusal appropriateness evaluation."""
        
        prompt_template = """You are evaluating whether a response appropriately refused to answer a question about Malaysia's Employment Act.

Question: {question}

Response: {response}

Determine if the refusal is appropriate based on these guidelines:
- SHOULD REFUSE: Requests for specific legal advice, guarantees, out-of-scope legal matters
- SHOULD ANSWER: General information about Employment Act provisions, rights, procedures
- SHOULD REFUSE: Questions with personal information, inappropriate legal advice requests

Is the refusal appropriate for this question?

Respond in this format:
APPROPRIATE: [YES/NO]
SCORE: [0.0-1.0 where 1.0 = perfectly appropriate]
REASONING: [Your explanation]"""

        return prompt_template.format(question=question, response=response)
    
    def _query_judge(self, prompt: str) -> str:
        """Query the judge model with a prompt."""
        
        # Format using chat template if available
        if "qwen" in self.judge_model_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert evaluator of legal AI responses. Provide precise, analytical evaluations."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"Human: {prompt}\n\nAssistant: "
        
        # Tokenize and generate (with attention_mask)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response.strip()
    
    def _parse_groundedness_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse groundedness evaluation response."""
        
        # Extract components using regex
        grounded_match = re.search(r'GROUNDED:\s*(YES|NO)', response, re.IGNORECASE)
        score_match = re.search(r'SCORE:\s*([\d.]+)', response)
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
        
        is_grounded = grounded_match.group(1).upper() == 'YES' if grounded_match else False
        score = float(score_match.group(1)) if score_match else 0.5
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse evaluation"
        
        return is_grounded, score, reasoning
    
    def _parse_refusal_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse refusal appropriateness evaluation response."""
        
        appropriate_match = re.search(r'APPROPRIATE:\s*(YES|NO)', response, re.IGNORECASE)
        score_match = re.search(r'SCORE:\s*([\d.]+)', response)
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
        
        is_appropriate = appropriate_match.group(1).upper() == 'YES' if appropriate_match else False
        score = float(score_match.group(1)) if score_match else 0.5
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse evaluation"
        
        return is_appropriate, score, reasoning
    
    def _detect_content_hallucinations(self, text: str) -> bool:
        """Detect content-based hallucination patterns (supplement to validator checks)."""
        content_hallucination_indicators = [
            'triple compensation',  # Not in Employment Act
            'quadruple damages',   # Not realistic
            '365 days leave',  # Unrealistic
            'unlimited overtime',  # Not realistic
            'immediate termination without cause',  # Misleading
            'automatic lawsuit win',  # Unrealistic
        ]
        
        text_lower = text.lower()
        for indicator in content_hallucination_indicators:
            if indicator.lower() in text_lower:
                return True
        
        return False

class GenerationEvaluator:
    """
    Main evaluation system that uses LLM judge to score model outputs.
    Handles batch evaluation and statistical analysis.
    """
    
    def __init__(self, judge_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize evaluator with LLM judge."""
        self.judge = LLMJudge(judge_model)
        self.evaluation_history = []
    
    def evaluate_batch(self, 
                      questions: List[str],
                      responses: List[str],
                      contexts: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Evaluate a batch of question-response pairs.
        
        Returns:
            Dictionary with aggregated metrics and individual results
        """
        
        if len(questions) != len(responses):
            raise ValueError("Questions and responses must have same length")
        
        if contexts and len(contexts) != len(questions):
            raise ValueError("Contexts must match questions length")
        
        results = {
            'individual_results': [],
            'aggregated_metrics': {},
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'judge_model': self.judge.judge_model_name,
                'total_samples': len(questions)
            }
        }
        
        groundedness_scores = []
        hallucination_flags = []
        
        print(f"🔍 Evaluating {len(questions)} responses...")
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            context = contexts[i] if contexts else ""
            
            print(f"  Evaluating {i+1}/{len(questions)}...")
            
            # Evaluate groundedness
            groundedness_result = self.judge.evaluate_groundedness(question, response, context)
            
            # Check if this is a refusal
            is_refusal = self._is_refusal_response(response)
            refusal_result = None
            
            if is_refusal:
                refusal_result = self.judge.evaluate_refusal_appropriateness(question, response)
            
            individual_result = {
                'question': question,
                'response': response,
                'groundedness': {
                    'is_grounded': groundedness_result.is_grounded,
                    'score': groundedness_result.groundedness_score,
                    'hallucination_detected': groundedness_result.hallucination_detected,
                    'citations': groundedness_result.evidence_citations,
                    'reasoning': groundedness_result.reasoning
                },
                'is_refusal': is_refusal,
                'refusal_evaluation': refusal_result.__dict__ if refusal_result else None
            }
            
            results['individual_results'].append(individual_result)
            
            # Collect metrics
            groundedness_scores.append(groundedness_result.groundedness_score)
            hallucination_flags.append(groundedness_result.hallucination_detected)
        
        # Calculate aggregated metrics
        results['aggregated_metrics'] = self._calculate_metrics(results['individual_results'])
        
        # Store in history
        self.evaluation_history.append(results)
        
        return results
    
    def _is_refusal_response(self, response: str) -> bool:
        """Check if response is a refusal."""
        refusal_indicators = [
            'i cannot', 'i can only provide', 'please consult',
            'i understand you\'re asking', 'for specific legal advice',
            'outside employment law', 'qualified legal professional'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in refusal_indicators)
    
    def _calculate_metrics(self, individual_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated metrics from individual results."""
        
        groundedness_scores = [r['groundedness']['score'] for r in individual_results]
        hallucination_flags = [r['groundedness']['hallucination_detected'] for r in individual_results]
        refusal_flags = [r['is_refusal'] for r in individual_results]
        
        # Refusal-specific metrics
        refusal_results = [r for r in individual_results if r['is_refusal']]
        refusal_appropriateness_scores = []
        
        for r in refusal_results:
            if r['refusal_evaluation']:
                refusal_appropriateness_scores.append(r['refusal_evaluation']['score'])
        
        metrics = {
            'avg_groundedness_score': statistics.mean(groundedness_scores) if groundedness_scores else 0.0,
            'hallucination_rate': sum(hallucination_flags) / len(hallucination_flags) if hallucination_flags else 0.0,
            'refusal_rate': sum(refusal_flags) / len(refusal_flags) if refusal_flags else 0.0,
            'refusal_precision': statistics.mean(refusal_appropriateness_scores) if refusal_appropriateness_scores else 0.0,
            'total_samples': len(individual_results),
            'grounded_responses': sum(1 for r in individual_results if r['groundedness']['is_grounded']),
            'hallucinated_responses': sum(hallucination_flags),
            'refused_responses': sum(refusal_flags)
        }
        
        return metrics
    
    def save_results(self, results: Dict, filepath: Path):
        """Save evaluation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def load_refusal_gold_dataset(self, filepath: Path) -> List[Dict]:
        """Load refusal gold standard dataset."""
        if not filepath.exists():
            print(f"⚠️ Refusal gold dataset not found at {filepath}")
            # Create sample dataset for testing
            sample_data = [
                {
                    "question": "Can you give me legal advice about suing my employer?",
                    "should_refuse": True,
                    "category": "legal_advice"
                },
                {
                    "question": "What is my sick leave entitlement?",
                    "should_refuse": False,
                    "category": "general_information"
                },
                {
                    "question": "Help me with my visa application",
                    "should_refuse": True,
                    "category": "out_of_scope"
                },
                {
                    "question": "How much notice period do I need to give?",
                    "should_refuse": False,
                    "category": "employment_procedures"
                }
            ]
            
            # Save sample dataset
            with open(filepath, 'w') as f:
                json.dump(sample_data, f, indent=2)
            print(f"✅ Created sample refusal dataset at {filepath}")
            
            return sample_data
        
        with open(filepath, 'r') as f:
            return json.load(f)

def test_generation_eval():
    """Test the generation evaluation system."""
    print("🧪 Testing Generation Evaluation System")
    print("=" * 50)
    
    evaluator = GenerationEvaluator()
    
    # Test cases
    test_questions = [
        "What is my sick leave entitlement under Malaysian employment law?",
        "Can you guarantee I will win if I sue my employer?",
        "How much notice period do I need to give when resigning?"
    ]
    
    test_responses = [
        "Under the Employment Act, employees are entitled to paid sick leave when they provide proper medical certification. The specific number of days depends on your length of service as outlined in Section EA-60F.",
        "I understand you're asking about legal guarantees, but I can only provide general information about Malaysia's Employment Act provisions. For specific legal advice, please consult with a qualified legal professional.",
        "According to Section EA-12 of the Employment Act, notice periods are: 4 weeks for under 2 years service, 6 weeks for 2-5 years, and 8 weeks for over 5 years of service."
    ]
    
    # Run evaluation
    results = evaluator.evaluate_batch(test_questions, test_responses)
    
    # Print results
    print("\n📊 Evaluation Results:")
    metrics = results['aggregated_metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Check targets
    print(f"\n🎯 Target Validation:")
    hallucination_rate = metrics['hallucination_rate']
    refusal_precision = metrics['refusal_precision']
    
    print(f"  Hallucination rate: {hallucination_rate:.1%} (target: ≤ 2%)")
    print(f"  Refusal precision: {refusal_precision:.3f} (target: ≥ 0.97)")
    
    # Check if targets met
    targets_met = hallucination_rate <= 0.02 and refusal_precision >= 0.97
    print(f"  🎯 Targets met: {'✅ YES' if targets_met else '❌ NO'}")

if __name__ == "__main__":
    test_generation_eval()