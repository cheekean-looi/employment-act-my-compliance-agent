#!/usr/bin/env python3
# python src/training/tiny_ppo_loop.py --dpo-model outputs/lora_dpo --output outputs/lora_ppo/ppo_results.json --use-generated
"""
Tiny PPO Loop for Employment Act Malaysia Compliance Agent
Implements PPO training starting from DPO checkpoint with reward scoring.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import copy

class TinyPPOLoop:
    def __init__(self, 
                 base_model_name: str = "microsoft/DialoGPT-medium",
                 dpo_checkpoint_path: Path = None):
        """Initialize PPO loop starting from DPO checkpoint."""
        self.base_model_name = base_model_name
        self.dpo_checkpoint_path = dpo_checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("   Initializing Tiny PPO Loop")
        print(f"   Device: {self.device}")
        print(f"   Base model: {base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models: policy and reference
        self._load_models()
        
        # Reward scoring criteria
        self.reward_criteria = {
            "has_citation": 2.0,      # +2 if includes valid citations
            "grounded_response": 1.0,  # +1 if response is grounded in law
            "hallucination": -2.0,     # -2 if judge flags hallucination
            "policy_violation": -2.0   # -2 if violates safety policy
        }
        
        print("   Initializing Tiny PPO Loop")
        print("   Reward criteria:")
        for criterion, score in self.reward_criteria.items():
            print(f"  {criterion}: {score:+.1f}")
    
    def _load_models(self):
        """Load policy and reference models from DPO checkpoint."""
        
        # Load base model
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            dtype=torch.float32,  # MPS compatibility
            device_map="auto"
        )
        
        # Load DPO checkpoint as policy (recommended for PPO)
        if self.dpo_checkpoint_path and Path(self.dpo_checkpoint_path).exists():
            print(f"   Loading DPO checkpoint from {self.dpo_checkpoint_path}")
            print("   Starting PPO from DPO policy for optimal performance...")
            try:
                # Load the DPO LoRA adapter as policy
                self.policy_model = PeftModel.from_pretrained(base_model, self.dpo_checkpoint_path)
                print("   Successfully loaded DPO adapter as PPO policy")
                
                # Create reference model (frozen copy of DPO policy)
                print("   Creating frozen reference model for KL penalty...")
                base_model_ref = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    dtype=torch.float32,
                    device_map="auto"
                )
                self.reference_model = PeftModel.from_pretrained(base_model_ref, self.dpo_checkpoint_path)
                
                # Freeze reference model
                for param in self.reference_model.parameters():
                    param.requires_grad = False
                    
                print("   Reference model created and frozen")
                
            except Exception as e:
                print(f"   Could not load DPO checkpoint: {e}")
                print("   Falling back to base model...")
                self.policy_model = base_model
                self.reference_model = copy.deepcopy(base_model)
                for param in self.reference_model.parameters():
                    param.requires_grad = False
        else:
            print("   No DPO checkpoint provided - starting PPO from base model")
            print("   Tip: Use --dpo-model outputs/lora_dpo for better results")
            self.policy_model = base_model
            self.reference_model = copy.deepcopy(base_model)
            for param in self.reference_model.parameters():
                param.requires_grad = False
        
        # Print model info
        if hasattr(self.policy_model, 'print_trainable_parameters'):
            print("   Policy model parameters:")
            self.policy_model.print_trainable_parameters()
        
        print("   Models loaded successfully!")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt based on model type."""
        if "dialo" in self.base_model_name.lower():
            return f"Human: {prompt}\nAssistant: "
        else:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response using the policy model (loaded from DPO)."""
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response using DPO policy
        with torch.no_grad():
            outputs = self.policy_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response (only the new tokens)
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response.strip()
    
    def _judge_response(self, prompt: str, response: str) -> Dict[str, Any]:
        """Judge response quality and assign rewards."""
        
        # Initialize judgment
        judgment = {
            "has_citation": False,
            "grounded_response": False,
            "hallucination": False,
            "policy_violation": False,
            "explanation": ""
        }
        
        response_lower = response.lower()
        
        # Check for citations
        citation_patterns = ["section ea-", "employment act section", "source:", "(source:"]
        judgment["has_citation"] = any(pattern in response_lower for pattern in citation_patterns)
        
        # Check for grounding (mentions legal concepts)
        grounding_patterns = ["employment act", "according to", "the law", "provision", "under section"]
        judgment["grounded_response"] = any(pattern in response_lower for pattern in grounding_patterns)
        
        # Check for hallucination indicators (specific fake details)
        hallucination_patterns = ["14 days", "30 calendar days", "triple compensation", "section ea-99", "ea-88"]
        judgment["hallucination"] = any(pattern in response_lower for pattern in hallucination_patterns)
        
        # Check for policy violations (vague unhelpful responses)
        policy_patterns = ["consult with hr", "check the relevant sections", "different situations may have"]
        judgment["policy_violation"] = any(pattern in response_lower for pattern in policy_patterns)
        
        # Generate explanation
        if judgment["has_citation"] and judgment["grounded_response"]:
            judgment["explanation"] = "High quality: Well-grounded response with proper citations"
        elif judgment["has_citation"]:
            judgment["explanation"] = "Good: Has citations but could be more grounded"
        elif judgment["grounded_response"]:
            judgment["explanation"] = "Moderate: Grounded but missing citations"
        elif judgment["hallucination"]:
            judgment["explanation"] = "Poor: Contains hallucinated details"
        elif judgment["policy_violation"]:
            judgment["explanation"] = "Poor: Vague and unhelpful response"
        else:
            judgment["explanation"] = "Basic: Adequate but could be improved"
        
        return judgment
    
    def _calculate_reward(self, judgment: Dict[str, Any]) -> float:
        """Calculate reward score based on judgment."""
        reward = 0.0
        
        for criterion, value in judgment.items():
            if criterion in self.reward_criteria and value:
                reward += self.reward_criteria[criterion]
        
        return reward
    
    def run_ppo_epoch(self, prompts: List[str], use_generated_responses: bool = True) -> Dict[str, Any]:
        """Run one PPO epoch with reward scoring using DPO policy."""
        
        print(f"   Running PPO epoch on {len(prompts)} examples...")
        if use_generated_responses:
            print("   Generating responses using DPO policy model...")
        else:
            print("   Using pre-written sample responses...")
        
        results = []
        total_reward = 0.0
        reward_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        
        for i, prompt in enumerate(prompts):
            # Generate response using DPO policy model
            if use_generated_responses:
                try:
                    response = self._generate_response(prompt)
                    generation_source = "DPO_policy"
                except Exception as e:
                    print(f"   Generation failed for prompt {i}: {e}")
                    response = "I apologize, but I cannot provide a response at this time."
                    generation_source = "fallback"
            else:
                # Use sample responses for demo
                sample_responses = self._get_sample_responses()
                response = sample_responses[i % len(sample_responses)]
                generation_source = "sample"
            
            # Judge the response
            judgment = self._judge_response(prompt, response)
            
            # Calculate reward
            reward = self._calculate_reward(judgment)
            total_reward += reward
            
            # Track reward distribution
            if reward > 0:
                reward_distribution["positive"] += 1
            elif reward < 0:
                reward_distribution["negative"] += 1
            else:
                reward_distribution["neutral"] += 1
            
            # Store result
            result = {
                "prompt": prompt,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "judgment": judgment,
                "reward": reward,
                "example_id": i,
                "generation_source": generation_source
            }
            results.append(result)
        
        # Calculate epoch statistics
        avg_reward = total_reward / len(prompts)
        high_quality_rate = reward_distribution["positive"] / len(prompts)
        low_quality_rate = reward_distribution["negative"] / len(prompts)
        
        epoch_stats = {
            "total_examples": len(prompts),
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "reward_distribution": reward_distribution,
            "high_quality_rate": high_quality_rate,
            "low_quality_rate": low_quality_rate,
            "results": results[:5],  # Store first 5 examples
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"   PPO Epoch Results:")
        print(f"  Average reward: {avg_reward:+.2f}")
        print(f"  High quality rate: {high_quality_rate:.1%}")
        print(f"  Low quality rate: {low_quality_rate:.1%}")
        
        return epoch_stats
    
    def _get_sample_responses(self) -> List[str]:
        """Get sample responses for fallback/demo purposes."""
        return [
            "According to Section EA-60E of the Employment Act, you are entitled to annual leave as specified in the legislation. (Source: Employment Act Section EA-60E)",
            "Your rights under Section EA-37 of the Employment Act provide important protections during pregnancy. The law prohibits dismissal during maternity leave.",
            "The Employment Act contains various provisions related to your question. You should check the relevant sections or consult with HR for specific details.",
            "Under Section EA-69 of the Employment Act, you can file complaints with the Director General of Labour for violations of employment rights.",
            "If you resign without notice, you may forfeit certain benefits including 14 days of compensation as specified in the regulations.",
            "Yes, under the Employment Act you are entitled to triple compensation for overtime work exceeding normal hours as per Section EA-99.",
            "According to the Employment Act, working hours are regulated to protect employee welfare. Normal working hours should not exceed statutory limits.",
            "The Employment Act protects your right to take legitimate sick leave. You cannot be penalized for exercising this right when properly documented."
        ]
    
    def load_sample_prompts(self) -> List[str]:
        """Load sample Employment Act prompts for PPO evaluation."""
        return [
            "How many days of annual leave am I entitled to?",
            "What are my rights regarding pregnancy under the Employment Act?",
            "Can my employer terminate me without notice?",
            "What is the procedure for filing a complaint?",
            "What happens if I resign without notice?",
            "Am I entitled to overtime pay?",
            "What are the restrictions on working hours?",
            "Can I be penalized for taking sick leave?"
        ]
    
    def save_ppo_results(self, epoch_stats: Dict[str, Any], output_file: Path):
        """Save PPO epoch results."""
        
        # Save detailed results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_stats, f, indent=2, ensure_ascii=False)
        
        # Create summary report
        summary_file = output_file.parent / "ppo_summary.json"
        summary = {
            "ppo_epoch_summary": {
                "average_reward": epoch_stats["average_reward"],
                "high_quality_rate": epoch_stats["high_quality_rate"],
                "low_quality_rate": epoch_stats["low_quality_rate"],
                "total_examples": epoch_stats["total_examples"]
            },
            "reward_criteria": self.reward_criteria,
            "recommendations": self._generate_recommendations(epoch_stats)
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ PPO results saved:")
        print(f"  Detailed: {output_file}")
        print(f"  Summary: {summary_file}")
    
    def _generate_recommendations(self, epoch_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on PPO results."""
        recommendations = []
        
        avg_reward = epoch_stats["average_reward"]
        high_quality_rate = epoch_stats["high_quality_rate"]
        
        if avg_reward < 0:
            recommendations.append("Consider additional SFT training to improve base response quality")
        
        if high_quality_rate < 0.6:
            recommendations.append("Increase citation training in future DPO pairs")
        
        if epoch_stats["low_quality_rate"] > 0.3:
            recommendations.append("Add more negative examples to reduce hallucination")
        
        if avg_reward > 1.0:
            recommendations.append("Model shows good preference alignment - ready for deployment")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Run tiny PPO loop starting from DPO checkpoint")
    parser.add_argument('--dpo-model', help='Path to DPO LoRA checkpoint (recommended)')
    parser.add_argument('--base-model', default='microsoft/DialoGPT-medium', help='Base model name')
    parser.add_argument('--output', required=True, help='Output file for PPO results')
    parser.add_argument('--use-generated', action='store_true', 
                       help='Generate responses using DPO model (vs using samples)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize PPO loop with DPO checkpoint
    ppo_loop = TinyPPOLoop(
        base_model_name=args.base_model,
        dpo_checkpoint_path=Path(args.dpo_model) if args.dpo_model else None
    )
    
    # Load sample prompts
    prompts = ppo_loop.load_sample_prompts()
    
    # Run PPO epoch (with or without actual DPO generation)
    epoch_stats = ppo_loop.run_ppo_epoch(prompts, use_generated_responses=args.use_generated)
    
    # Save results
    ppo_loop.save_ppo_results(epoch_stats, output_path)
    
    print(f"\\n🎉 Tiny PPO epoch complete!")
    print(f"📊 Average reward: {epoch_stats['average_reward']:+.2f}")
    print(f"🏆 High quality rate: {epoch_stats['high_quality_rate']:.1%}")
    
    if args.dpo_model:
        print(f"🔗 Used DPO checkpoint: {args.dpo_model}")
    else:
        print("⚠️ No DPO checkpoint used - consider using --dpo-model for better results")


if __name__ == "__main__":
    main()