SCREENING_PROMPT_TEMPLATE = """
# CONTEXT #
You are a clinical specialist tasked with assessing research papers for inclusion in a systematic literature review based on specific eligibility criteria.

# OBJECTIVE #
Evaluate each criterion of a given paper to determine its eligibility for inclusion in the review. Provide a list of decisions ("YES", "PARTIAL", "NO", or "UNCERTAIN") for each eligibility criterion. You must deliver exactly {num_criteria} responses.
1. YES: Meets the criteria.
2. PARTIAL: Partially meets the criteria but not completely.
3. NO: Does not meet the criteria.
4. UNCERTAIN: Uncertain if it meets the criteria.

# IMPORTANT NOTE #
If the information within the provided paper content is insufficient to conclusively evaluate a criterion, you must opt for "UNCERTAIN" as your response. Avoid making assumptions or extrapolating beyond the provided data, as accurate and reliable responses are crucial, and fabricating information (hallucinations) could lead to serious errors in the systematic review.
If the information is not applicable N/A, you also must opt for "UNCERTAIN".
Use "PARTIAL" when the paper meets some aspects of the criterion but not all; ensure that the partial fulfillment is based on the provided data and not on assumptions or incomplete information.

# PAPER DETAILS #
- Provided Paper: {paper_content}

# EVALUATION CRITERIA #
- Number of Criteria: {num_criteria}
- Criteria for Inclusion: {criteria_text}

# RESPONSE #
You are required to output a JSON object containing a list of decisions for each of the {num_criteria} eligibility criteria. Each decision should directly correspond to one of the criteria and be listed in the order they are presented. Ensure to use "UNCERTAIN" wherever the paper does not explicitly support a "YES", "PARTIAL", or "NO" decision.
The length of "evaluation" should be exactly {num_criteria}.
For example:
```json
{{
    "evaluations": [ \\ list of eligibility decisions for each criterion
        {{
            "eligibility": "YES", \\ decision for the first criterion
            "rationale": "..." \\ rationale for the decision, limited in 50 tokens
        }},
        {{
            "eligibility": "PARTIAL", \\ decision for the second criterion
            "rationale": "..." \\ rationale for the decision, limited in 50 tokens
        }},
        {{
            "eligibility": "NO", \\ decision for the third criterion
            "rationale": "..." \\ rationale for the decision, limited in 50 tokens
        }},
        {{
            "eligibility": "UNCERTAIN", \\ decision for the fourth criterion
            "rationale": "..." \\ rationale for the decision, limited in 50 tokens
        }},
        ... \\ continue for all criteria
    ]
}}
```
"""

import pdb
import os
import re
import json
from ..client import call_leads
import tiktoken

def cut_paper_content(paper_content, max_tokens=29_000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(paper_content)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        paper_content = encoding.decode(tokens)
    return paper_content

def extract_json_from_llm_output(text):
    """Extract eligibility predictions from LLM output text, with multiple fallback methods."""
    
    try:
        json_obj = json.loads(text)
        if json_obj.get('evaluations'):
            return json_obj
    except json.JSONDecodeError:
        pass

    # Try to find and parse JSON within code blocks first
    pattern_code_block = r"```(?:json)?\n([\s\S]+?)\n```"
    if match := re.search(pattern_code_block, text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find and parse any JSON-like structure
    pattern_json = r"\{[\s\S]*?\}"
    if match := re.search(pattern_json, text):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If JSON parsing fails, try to extract individual eligibility predictions
    evaluations = []
    
    # Pattern for individual evaluations
    pattern_eval = r"(?:\"eligibility\":\s*\"(YES|NO|UNCERTAIN)\"[\s,]*\"rationale\":\s*\"([^\"]+)\")|(?:eligibility:\s*(YES|NO|UNCERTAIN)[\s,]*rationale:\s*([^,\}]+))"
    
    matches = re.finditer(pattern_eval, text)
    for match in matches:
        # Handle both JSON and non-JSON formatted matches
        if match.group(1) and match.group(2):  # JSON format
            evaluations.append({
                "eligibility": match.group(1),
                "rationale": match.group(2).strip()
            })
        elif match.group(3) and match.group(4):  # Non-JSON format
            evaluations.append({
                "eligibility": match.group(3),
                "rationale": match.group(4).strip()
            })
    
    if evaluations:
        return {"evaluations": evaluations}
    
    # If no structured data found, try to extract just eligibility decisions
    pattern_simple = r"(?:eligibility|decision|result):\s*(YES|NO|UNCERTAIN)"
    matches = re.finditer(pattern_simple, text, re.IGNORECASE)
    evaluations = [
        {
            "eligibility": match.group(1).upper(),
            "rationale": "No explicit rationale provided"
        }
        for match in matches
    ]
    
    if evaluations:
        return {"evaluations": evaluations}
    
    return {"evaluations": []}

def get_eligibility_criteria(PICO):
    pico_name_map = {
        "P": "Population",
        "I": "Intervention",
        "C": "Comparison",
        "O": "Outcome"
    }
    selected_pico = ["P", "I", "C", "O"]
    criteria = [f"{pico_name_map[p]}: {PICO[p]}" for p in selected_pico]
    num_criteria = len(criteria)
    return criteria, num_criteria

def stringfy_criteria(criteria):
    criteria_text = ". ".join(f"{i+1}: {c}" for i, c in enumerate(criteria))
    return criteria_text

# get overall score based on evaluations of each criteria
def get_score(evaluations):
    score = 0
    for eval in evaluations:
        if eval['eligibility'] == 'YES':
            score += 1
        elif eval['eligibility'] == 'PARTIAL':
            score += 0.5
        elif eval['eligibility'] == 'UNCERTAIN':
            score += 0
        elif eval['eligibility'] == 'NO':
            score -= 1
    return score / len(evaluations)

def screening_study(paper_content, population=None, intervention=None, comparison=None, outcome=None):
    """
    Perform screening study on a given paper content.

    Args:
        paper_content (str): The content of the paper to be screened.
        population (str): The population of the research
        intervention (str): The intervention of the research
        comparison (str): The comparison of the research
        outcome (str): The outcome of the research
    """
    assert population or intervention or comparison or outcome, "At least one of population, intervention, comparison, or outcome must be provided."
    PICO = {
        "P": population if population else "",
        "I": intervention if intervention else "",
        "C": comparison if comparison else "",
        "O": outcome if outcome else ""
    }
    criteria, num_criteria = get_eligibility_criteria(PICO)
    criteria_text = stringfy_criteria(criteria)
    paper_content = cut_paper_content(paper_content)
    prompt = SCREENING_PROMPT_TEMPLATE.format(paper_content=paper_content, num_criteria=num_criteria, criteria_text=criteria_text)
    results = call_leads(
        prompt, 
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"),
        temperature=0.1,
        max_tokens=1024
    )
    # parse the results
    evaluations = extract_json_from_llm_output(results)
    evaluations = evaluations.get('evaluations', [])
    if len(evaluations) == 0:
        # make uncertain for all criteria
        evaluations = [{"eligibility": "UNCERTAIN", "rationale": "No eligibility predictions found in the text."} for _ in range(num_criteria)]
    score = get_score(evaluations)
    return evaluations, score

def batch_screening_study(paper_contents, population=None, intervention=None, comparison=None, outcome=None):
    """
    Perform screening study on a list of paper contents.

    Args:
        paper_contents (list): A list of paper contents to be screened.
        population (str): The population of the research
        intervention (str): The intervention of the research
        comparison (str): The comparison of the research
        outcome (str): The outcome of the research
    Returns:
        list: A list of screening results.
    """
    assert population or intervention or comparison or outcome, "At least one of population, intervention, comparison, or outcome must be provided."
    PICO = {
        "P": population if  population else "",
        "I": intervention if  intervention else "",
        "C": comparison if  comparison else "",
        "O": outcome if  outcome else ""
    }
    criteria, num_criteria = get_eligibility_criteria(PICO)
    criteria_text = stringfy_criteria(criteria)
    paper_contents = [cut_paper_content(paper_content) for paper_content in paper_contents]
    prompts = [SCREENING_PROMPT_TEMPLATE.format(paper_content=paper_content, num_criteria=num_criteria, criteria_text=criteria_text) for paper_content in paper_contents]
    results = call_leads(
        prompts, 
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"),
        temperature=0.1,
        max_tokens=1024
    )
    # parse the results
    evaluations = [extract_json_from_llm_output(result) for result in results]
    evaluations = [eval.get('evaluations', []) for eval in evaluations]
    scores = [get_score(eval) for eval in evaluations]
    tuple_results = list(zip(evaluations, scores))
    return tuple_results
