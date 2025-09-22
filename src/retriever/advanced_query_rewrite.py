"""
Advanced query rewriter with statutory term mapping for Employment Act Malaysia.
Maps user language to precise legal terminology used in the Act.
"""

import re
from typing import List, Dict, Tuple

class AdvancedQueryRewriter:
    def __init__(self):
        # Canonical statutory term mappings
        self.statutory_terms = {
            # Employment termination
            r"\bseverance\b|\bretrench(ment)?\b|\breadundan(cy|t)\b": "termination and lay-off benefits",
            r"\bmisconduct\b": "misconduct dismissal just cause",
            r"\bconstructive dismissal\b": "constructive dismissal termination",
            r"\bresign without notice\b": "resignation without notice termination",
            r"\bterminate.*pregnancy\b|\bpregnant.*terminate\b": "pregnant employee termination protection",
            
            # Leave and benefits
            r"\bpregnan(t|cy)\b": "pregnancy maternity confinement pregnant employee",
            r"\bmaternity leave\b": "maternity confinement benefit",
            r"\bpaternity leave\b": "paternity benefit",
            r"\bannual leave\b": "annual leave vacation",
            r"\bsick leave\b": "sick leave medical absence",
            r"\bemergency leave\b": "emergency leave compassionate",
            
            # Working conditions
            r"\bworking hours\b": "hours of work normal hours",
            r"\bovertime pay\b": "overtime payment rate of pay",
            r"\brest day(s)?\b": "rest day weekly rest",
            r"\bpublic holiday(s)?\b": "public holiday gazetted holiday",
            
            # Compensation
            r"\bbonus(es)?\b": "bonus additional payment incentive",
            r"\bsalary\b|\bwage(s)?\b": "wages payment of wages",
            r"\bdeduct(ion)?\b": "deduction from wages authorized deduction",
            
            # Employment terms
            r"\bprobation(ary)?\b|\btrial period\b": "probationary period trial employment contract",
            r"\bretirement age\b|\bminimum retirement\b": "minimum retirement age compulsory retirement mandatory",
            r"\bnotice period\b": "period of notice termination notice",
            r"\bfemale employee\b": "female employee woman worker",
            r"\bcomplaint\b|\bgrievance\b": "complaint inquiry investigation",
            r"\bbonus.*entitle\b|\bentitle.*bonus\b": "additional payment bonus statutory entitlement",
        }
        
        # Section-specific keywords for metadata filtering
        self.section_keywords = {
            "maternity": ["EA-37", "EA-38", "EA-40", "EA-42", "EA-44"],
            "pregnancy": ["EA-40", "EA-42", "EA-44"],
            "pregnant": ["EA-40", "EA-42", "EA-44"],
            "overtime": ["EA-60A", "EA-62"],
            "public holiday": ["EA-60D", "EA-60I"],
            "annual leave": ["EA-60E"],
            "sick leave": ["EA-60F"],
            "termination": ["EA-12", "EA-13", "EA-14"],
            "wages": ["EA-25", "EA-25A", "EA-27"],
            "working hours": ["EA-60", "EA-60A"],
            "rest day": ["EA-60A", "EA-61"],
            "probation": ["EA-11"],
            "probationary": ["EA-11"],
            "trial period": ["EA-11"],
            "bonus": ["EA-25A"],
            "additional payment": ["EA-25A"],
            "retirement": ["EA-11"],
            "retirement age": ["EA-11"],
            "minimum retirement": ["EA-11"],
        }
        
        # Out-of-scope queries (no statutory basis)
        self.out_of_scope_patterns = [
            r"\bcontractual bonus\b",
            r"\bperformance bonus\b", 
            r"\bincentive scheme\b",
            r"\bcommission\b",
        ]
    
    def normalize_query(self, query: str) -> Tuple[str, List[str], bool]:
        """
        Normalize query to statutory terms.
        
        Returns:
            - normalized_query: Query with statutory terms
            - priority_sections: Sections likely to contain answer
            - is_out_of_scope: Whether query has no statutory basis
        """
        normalized = query.lower()
        priority_sections = []
        is_out_of_scope = False
        
        # Check for out-of-scope patterns first
        for pattern in self.out_of_scope_patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                is_out_of_scope = True
                break
        
        # Apply statutory term substitutions
        for pattern, replacement in self.statutory_terms.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Extract priority sections based on topic
        for topic, sections in self.section_keywords.items():
            if topic in normalized:
                priority_sections.extend(sections)
        
        return normalized, list(set(priority_sections)), is_out_of_scope
    
    def expand_query(self, query: str) -> List[str]:
        """Generate multiple query variants for comprehensive retrieval."""
        normalized, priority_sections, is_out_of_scope = self.normalize_query(query)
        
        if is_out_of_scope:
            return [query]  # Don't expand out-of-scope queries
        
        variants = [query, normalized]  # Original + normalized
        
        # Add Employment Act context
        variants.append(f"Employment Act Malaysia {normalized}")
        
        # Add section-specific variants if we have priority sections
        if priority_sections:
            section_str = " ".join(priority_sections[:3])  # Top 3 sections
            variants.append(f"{normalized} section {section_str}")
        
        # Add legal context variants
        legal_variants = [
            f"statutory {normalized}",
            f"legal requirements {normalized}",
            f"{normalized} provisions law"
        ]
        variants.extend(legal_variants)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant.lower() not in seen:
                unique_variants.append(variant)
                seen.add(variant.lower())
        
        return unique_variants[:6]  # Limit to 6 variants
    
    def get_failure_category(self, query: str, retrieved_sections: List[str]) -> str:
        """Categorize why a query might have failed."""
        normalized, priority_sections, is_out_of_scope = self.normalize_query(query)
        
        if is_out_of_scope:
            return "OUT_OF_SCOPE"
        
        if not priority_sections:
            return "COVERAGE_MISS"
        
        # Check if any priority sections were retrieved
        if any(section in retrieved_sections for section in priority_sections):
            return "RANKING_MISS"  # Found but ranked too low
        else:
            return "VOCAB_MISS"  # Terms didn't match


def create_enhanced_retrieval_function():
    """Create enhanced retrieval function with advanced query rewriting."""
    rewriter = AdvancedQueryRewriter()
    
    def enhanced_retrieve(query: str, retriever, top_k: int = 8) -> Dict:
        """Enhanced retrieval with advanced query rewriting."""
        
        # Normalize and expand query
        normalized, priority_sections, is_out_of_scope = rewriter.normalize_query(query)
        
        if is_out_of_scope:
            return {
                "results": [],
                "should_refuse": True,
                "reason": "No statutory basis - contractual matter"
            }
        
        query_variants = rewriter.expand_query(query)
        
        # Collect candidates from all variants
        all_candidates = set()
        for variant in query_variants:
            try:
                # Use enhanced hybrid search
                results = retriever.retrieve(variant, top_k=top_k, use_rewrite=False)
                for result in results:
                    chunk_id = result.get('chunk_id', result.get('id', ''))
                    all_candidates.add(chunk_id)
            except Exception as e:
                continue
        
        # Re-rank all candidates using original query
        if hasattr(retriever, '_rerank_candidates'):
            final_results = retriever._rerank_candidates(
                query, 
                list(all_candidates), 
                top_k=top_k
            )
        else:
            final_results = list(all_candidates)[:top_k]
        
        # Boost priority sections if found
        if priority_sections:
            boosted_results = []
            non_priority = []
            
            for result in final_results:
                section_id = result.get('section_id', '')
                if section_id in priority_sections:
                    result['score'] = result.get('score', 0) + 1.0  # Boost score
                    boosted_results.append(result)
                else:
                    non_priority.append(result)
            
            # Combine: priority first, then others
            final_results = boosted_results + non_priority
        
        return {
            "results": final_results[:top_k],
            "should_refuse": False,
            "query_variants": query_variants,
            "priority_sections": priority_sections
        }
    
    return enhanced_retrieve