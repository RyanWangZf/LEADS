RESULT_EXTRACTION_PROMPT_TEMPLATE =  """
# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data.

# PAPER CONTENT #
{paper_content}

# TARGET #
Extract the results related to the specified outcome and group as follows:

Outcome: {outcome_def}

Group: {group_def}

# RESPONSE #
A syntactically correct JSON string:

Format:
```json
{{
    "paramType": \\ str, the type of the parameters
    "unitOfMeasure": \\ str, the unit of the result values
    "timeFrame": \\ str, the timeframe
    "unitOfDenom": \\ str, the unit of the denomintor for this group
    "denomValue": \\ int, the value of the group's denominator
    "results": [ \\ list of result values
        {{
            "value": int or float \\ the result value
            "title": str \\ the title for this value
        }},... \\ more results, if applicable
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
    return json.loads(llm_output)


def extract_trial_result(paper_content, outcome_def, group_def):
    paper_content = cut_paper_content(paper_content)
    prompt = RESULT_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content, outcome_def=outcome_def, group_def=group_def)
    results = call_leads(prompt,
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024
        )
    results = parse_llm_output(results)
    return results


def batch_extract_trial_result(paper_contents, outcome_def, group_def):
    paper_contents = [cut_paper_content(paper_content) for paper_content in paper_contents]
    prompts = [RESULT_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content, outcome_def=outcome_def, group_def=group_def) for paper_content in paper_contents]
    results = call_leads(
        prompts,
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024
        )
    results = [parse_llm_output(result) for result in results]
    return results