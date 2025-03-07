SEARCH_PROMPT_TEMPLATE = """
# CONTEXT #
You are a clinical specialist. You are conducting research and doing a medical literature review.

# OBJECTIVE #
Your task is to create query terms for a search URL to find relevant literature on PubMed or ClinicalTrials.gov.

# RESEARCH DETAILS #
The research is defined by the following PICO elements:
P (Patient, Problem or Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

# RESPONSE #
Your output should be in the following JSON format:
{{
    "query": "...."
}}
"""

import pdb
import os
import json
import re
from ..client import call_leads

def split_medical_query(query):
    """Split a complex medical query into manageable segments while preserving its logical structure."""
    
    # First, identify the main AND parts
    def split_on_main_and(query):
        # Find the main AND that separates the two major groups
        parts = []
        current_part = ""
        paren_count = 0
        i = 0
        
        while i < len(query):
            if query[i] == '(':
                paren_count += 1
                current_part += query[i]
            elif query[i] == ')':
                paren_count -= 1
                current_part += query[i]
            # Look for " AND " at the top level (when paren_count == 0)
            elif (paren_count == 0 and 
                  i + 5 < len(query) and 
                  query[i:i+5] == " AND "):
                parts.append(current_part.strip())
                current_part = ""
                i += 4  # Skip "AND"
            else:
                current_part += query[i]
            i += 1
            
        if current_part.strip():
            parts.append(current_part.strip())
        return parts

    def split_or_group(group, max_terms=5):
        """Split a group of OR terms into smaller subgroups"""
        # Remove outer parentheses
        group = group.strip()
        if group.startswith('(') and group.endswith(')'):
            group = group[1:-1].strip()
            
        terms = []
        current_term = ""
        paren_count = 0
        i = 0
        
        while i < len(group):
            if group[i] == '(':
                paren_count += 1
                current_term += group[i]
            elif group[i] == ')':
                paren_count -= 1
                current_term += group[i]
            # Look for " OR " at the top level
            elif (paren_count == 0 and 
                  i + 4 < len(group) and 
                  group[i:i+4] == " OR "):
                terms.append(current_term.strip())
                current_term = ""
                i += 3  # Skip "OR"
            else:
                current_term += group[i]
            i += 1
            
        if current_term.strip():
            terms.append(current_term.strip())
            
        # Split into subgroups
        subgroups = []
        current_subgroup = []
        
        for term in terms:
            current_subgroup.append(term)
            if len(current_subgroup) >= max_terms:
                subgroups.append(current_subgroup)
                current_subgroup = []
                
        if current_subgroup:
            subgroups.append(current_subgroup)
            
        return subgroups

    def construct_subqueries(condition_groups, intervention_groups):
        """Construct valid subqueries from combinations of condition and intervention groups"""
        subqueries = []
        
        for cond_group in condition_groups:
            for int_group in intervention_groups:
                subquery = f"({' OR '.join(cond_group)}) AND ({' OR '.join(int_group)})"
                subqueries.append(subquery)
                
        return subqueries

    # Split the main query into condition and intervention parts
    main_parts = split_on_main_and(query)
    if len(main_parts) != 2:
        raise ValueError("Query must have exactly two main parts connected by AND")
        
    conditions_part, interventions_part = main_parts
    
    # Split each part into smaller groups
    condition_groups = split_or_group(conditions_part)
    intervention_groups = split_or_group(interventions_part)
    
    # Construct the final subqueries
    return construct_subqueries(condition_groups, intervention_groups)


def parse_search_query(response):
    """
    Parse the search query from the response using regex as a fallback.
    
    Args:
        response (str): The response from the LLM containing a search query
        
    Returns:
        str: The extracted search query
    """
    # First try standard JSON parsing
    try:
        return json.loads(response)["query"]
    except (json.JSONDecodeError, KeyError):
        # Fallback to regex if JSON parsing fails
        query_pattern = r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"|"query"\s*:\s*([^"}\s].*?[^"}\s])(?=\s*[,}])'
        
        match = re.search(query_pattern, response)
        if match:
            # Return the first non-empty group
            return match.group(1) if match.group(1) else match.group(2)
        
        # If no match with the key, try to extract anything that looks like a PubMed query
        pubmed_pattern = r'\(\([^()]+\)\s+(?:AND|OR)\s+\([^()]+\)\)'
        match = re.search(pubmed_pattern, response)
        if match:
            return match.group(0)
            
        # If all else fails, return the original response
        return response

def search_query_generation(population=None, intervention=None, comparison=None, outcome=None):
    """
    Generate a search query for a given PICO elements.

    Args:
        population (str): The population of the research
        intervention (str): The intervention of the research
        comparison (str): The comparison of the research
        outcome (str): The outcome of the research
    """
    assert population or intervention or comparison or outcome, "At least one of population, intervention, comparison, or outcome must be provided."
    prompt = SEARCH_PROMPT_TEMPLATE.format(P=population, I=intervention, C=comparison, O=outcome)
    response = call_leads(prompt, endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), api_key=os.getenv("LEADS_API_KEY", "testtoken"))
    parsed_query = parse_search_query(response)
    sub_queries = split_medical_query(parsed_query)
    return sub_queries

def batch_search_query_generation(pico_list, batch_size=20):
    """
    Generate search queries for a list of PICO elements.

    Args:
        pico_list (list): A list of dictionaries, each containing the following keys:
            - "population" (str): The population of the research
            - "intervention" (str): The intervention of the research
            - "comparison" (str): The comparison of the research
            - "outcome" (str): The outcome of the research
    """
    all_prompts = [SEARCH_PROMPT_TEMPLATE.format(P=pico["population"], I=pico["intervention"], C=pico["comparison"], O=pico["outcome"]) for pico in pico_list]
    all_results = call_leads(all_prompts, endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), api_key=os.getenv("LEADS_API_KEY", "testtoken"), batch_size=batch_size)
    parsed_results = [parse_search_query(result) for result in all_results]
    sub_queries = [split_medical_query(parsed_result) for parsed_result in parsed_results]
    return sub_queries
