PARTICIPANT_EXTRACTION_PROMPT_TEMPLATE =  """
# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data.

# PAPER CONTENT #
{paper_content}

# TARGET #
Given the following information:
Parameter Type: {paramType}
Unit of Measurement: {unitOfMeasure}
Participant Groups Definition: {groupDef}

Where:
"groupId" is the group identifier
"unit" is the unit of measurement for the group
"value" is the numerical value representing the group's characteristic
"def" is the definition or description of the group

Extract the target participant's characteristics about {measureDef}.

# RESPONSE #
A syntactically correct JSON string:

Format:
```json
{{
    "results": [ \\ the list of characteristics for the groups
        {{
            "groupId": \\ str
            "value": \\ float or int, the value for the group
            "note": \\ str, the note for the value
        }},
        ... \\ more results
    ]
}}
```
"""

import json
import re
import pdb
import os
import tiktoken
from ..client import call_leads

def cut_paper_content(paper_content, max_tokens=29_000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(paper_content)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        paper_content = encoding.decode(tokens)
    return paper_content


def parse_llm_output(llm_output):
    """Parse LLM output to extract structured data with flexible formats.
    
    Args:
        llm_output (str): The LLM output text
        
    Returns:
        dict: Dictionary with extracted data maintaining original structure
    """
    # First try direct JSON parsing
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    # If JSON parsing fails, try to extract using regex patterns
    results = []
    
    # Pattern for population statistics format
    pop_pattern = r'{\s*"groupId":\s*"([^"]+)",\s*"value":\s*(\d+\.?\d*),\s*"note":\s*"([^"]+)"\s*}'
    
    # Pattern for trial results format
    trial_pattern = r'{\s*"value":\s*(\d+\.?\d*),\s*"title":\s*"([^"]*)"\s*}'
    
    # Try population statistics format first
    matches = re.finditer(pop_pattern, llm_output)
    for match in matches:
        results.append({
            "groupId": match.group(1),
            "value": float(match.group(2)),
            "note": match.group(3)
        })
    
    if results:
        return {"results": results}
    
    # Try trial results format
    results = []
    matches = re.finditer(trial_pattern, llm_output)
    for match in matches:
        results.append({
            "value": float(match.group(1)),
            "title": match.group(2)
        })
    
    if results:
        # Try to extract other fields if present
        param_type_match = re.search(r'"paramType":\s*"([^"]+)"', llm_output)
        unit_measure_match = re.search(r'"unitOfMeasure":\s*"([^"]+)"', llm_output)
        time_frame_match = re.search(r'"timeFrame":\s*"([^"]+)"', llm_output)
        unit_denom_match = re.search(r'"unitOfDenom":\s*"([^"]+)"', llm_output)
        denom_value_match = re.search(r'"denomValue":\s*(\d+\.?\d*)', llm_output)
        
        return {
            "paramType": param_type_match.group(1) if param_type_match else None,
            "unitOfMeasure": unit_measure_match.group(1) if unit_measure_match else None,
            "timeFrame": time_frame_match.group(1) if time_frame_match else None,
            "unitOfDenom": unit_denom_match.group(1) if unit_denom_match else None,
            "denomValue": float(denom_value_match.group(1)) if denom_value_match else None,
            "results": results
        }
    
    return {"results": []}

def extract_population_statistics(paper_content, measureDef, paramType, unitOfMeasure, groupDef):
    paper_content = cut_paper_content(paper_content)
    prompt = PARTICIPANT_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content, measureDef=measureDef, paramType=paramType, unitOfMeasure=unitOfMeasure, groupDef=groupDef)
    results = call_leads(prompt,
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024
    )
    results = parse_llm_output(results)
    return results

def batch_extract_population_statistics(paper_contents, measureDef, paramType, unitOfMeasure, groupDef):
    paper_contents = [cut_paper_content(paper_content) for paper_content in paper_contents]
    prompts = [PARTICIPANT_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content, measureDef=measureDef, paramType=paramType, unitOfMeasure=unitOfMeasure, groupDef=groupDef) for paper_content in paper_contents]
    results = call_leads(
        prompts,
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024
    )
    results = [parse_llm_output(result) for result in results]
    return results