STUDY_CHARACTERISTICS_EXTRACTION_PROMPT_TEMPLATE = """
# CONTEXT #
You are tasked with analyzing clinical trial study reports or papers to extract specific information as structured data.

# OBJECTIVE #
The user will provide a list of fields they are interested in, along with a natural language description for each field to guide you on what content to look for and from which parts of the report to extract it.

# PAPER CONTENT #
{paper_content}

# TARGET #
- Number of fields: {num_fields}
- Field definition: {fields_info}

# RESPONSE #
A syntactically correct JSON string representing a list of dictionary with two keys: name and value.
Format:
```json
[
    {{
        "name":  \\ str
        "value":  \\ str
    }},
    {{
        "name":, \\ str
        "value":  \\ str
    }},
    ...
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

def stringfy_fields_info(fields_info):
    fields_info_str_list = []
    for idx, field in enumerate(fields_info):
        fields_info_str_list.append(f"<field id={idx}>{field}</field>")
    fields_info_str = "\n".join(fields_info_str_list)
    return fields_info_str, len(fields_info_str_list)

def extract_json_from_llm_output(text, num_fields):
    """Extract the extraction results from LLM output text.
    
    Args:
        text (str): The LLM output text
        num_fields (int): Expected number of fields to extract
        
    Returns:
        dict: Dictionary with 'fields' key containing list of extracted fields
    """
    # First try direct JSON parsing
    try:
        json_obj = json.loads(text)
        if isinstance(json_obj, list):
            return {"fields": json_obj}
        if isinstance(json_obj.get('fields'), list):
            return json_obj
    except json.JSONDecodeError:
        pass

    # If JSON parsing fails, try to extract field dictionaries using regex
    fields = []
    pattern = r'{\s*"name":\s*"([^"]+)",\s*"value":\s*([^}]+)}'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        name = match.group(1).strip()
        value = match.group(2).strip().strip('"')
        
        # Try to convert numeric values
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except (ValueError, TypeError):
            pass
            
        fields.append({"name": name, "value": value})
    
    return {"fields": fields} if fields else {"fields": [{"name": f"Field {i+1}", "value": "Not found"} for i in range(num_fields)]}

def extract_study_characteristics(paper_content, fields_info):
    """Extract study characteristics from a paper content.

    Args:
        paper_content (str): The content of the paper to be screened.
        fields_info (list[str]): The fields information to be extracted.
    """
    fields_info_str, num_fields = stringfy_fields_info(fields_info)
    paper_content = cut_paper_content(paper_content)
    prompt = STUDY_CHARACTERISTICS_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content, num_fields=num_fields, fields_info=fields_info_str)
    results = call_leads(
        prompt, 
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024
        )
    # parse the results
    results = extract_json_from_llm_output(results, num_fields)
    return results


def batch_extract_study_characteristics(paper_contents, fields_info):
    """Batch extract study characteristics from a list of paper contents.

    Args:
        paper_contents (list[str]): The contents of the papers to be screened.
        fields_info (list[str]): The fields information to be extracted.
    """
    fields_info_str, num_fields = stringfy_fields_info(fields_info)
    paper_contents = [cut_paper_content(paper_content) for paper_content in paper_contents]
    prompts = [STUDY_CHARACTERISTICS_EXTRACTION_PROMPT_TEMPLATE.format(paper_content=paper_content, num_fields=num_fields, fields_info=fields_info_str) for paper_content in paper_contents]
    results = call_leads(
        prompts, 
        endpoint=os.getenv("LEADS_ENDPOINT", "http://localhost:13141/v1"), 
        api_key=os.getenv("LEADS_API_KEY", "testtoken"), 
        temperature=0.1, 
        max_tokens=1024
    )
    results = [extract_json_from_llm_output(result, num_fields) for result in results]
    return results