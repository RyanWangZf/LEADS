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

import os
from leads.client import call_leads

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
    return response

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
    all_prompts = []
    for pico in pico_list:
        prompt = SEARCH_PROMPT_TEMPLATE.format(P=pico["population"], I=pico["intervention"], C=pico["comparison"], O=pico["outcome"])
        all_prompts.append(prompt)
    return call_leads(all_prompts, endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), api_key=os.getenv("LEADS_API_KEY", "testtoken"), batch_size=batch_size)