ARM_DESIGN_EXTRACTION_PROMPT_TEMPLATE = """
# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data.

# PAPER CONTENT #
{paper_content}

# TARGET #
Extract the arm design of the study.

# RESPONSE #
A syntactically correct JSON string:

Format:
```json
[ \\ a list of arms
    {{
    "label": \\ str, the arm label
    "type": \\ str, the arm type
    "description": \\ str, the description of the arm
    "interventionNames": \\ list of str, the interventions used in this arm
    }},
    ... \\ more arms
]
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
    """Parse LLM output to extract list of dictionaries with arm label, type, description, and intervention names.
    
    Args:
        llm_output (str): The LLM output text
        
    Returns:
        list: A list of dictionaries with arm label, type, description, and intervention names.
    """
    # First try direct JSON parsing
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    # If JSON parsing fails, use regex to extract individual dictionaries
    results = []
    pattern = r'{\s*"label":\s*"([^"]+)",\s*"type":\s*"([^"]+)",\s*"description":\s*"([^"]+)",\s*"interventionNames":\s*(\[\s*"[^"]*",?\s*\])\s*}'
    matches = re.finditer(pattern, llm_output)
    for match in matches:
        arm = {
            "label": match.group(1),
            "type": match.group(2),
            "description": match.group(3),
            "interventionNames": json.loads(match.group(4))
        }
        results.append(arm)

    results = {
        "arms": results
    }
    
    return results

def extract_arm_design(paper_content):
    paper_content = cut_paper_content(paper_content)
    prompt = ARM_DESIGN_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content)
    results = call_leads(prompt,
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024)
    results = parse_llm_output(results)
    return results

def batch_extract_arm_design(paper_contents):
    paper_contents = [cut_paper_content(paper_content) for paper_content in paper_contents]
    prompts = [ARM_DESIGN_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content) for paper_content in paper_contents]
    results = call_leads(
        prompts,
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024
        )
    results = [parse_llm_output(result) for result in results]
    return results




