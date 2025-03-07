import os
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm

async def call_llm_single(prompt, pmid, client, model="zifeng-ai/leads-mistral-7b-v1", temperature=1.0, max_tokens=1024):
    """Make a single async call to the LLM."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {"pmid": pmid, "response": response.choices[0].message.content}
    except Exception as e:
        print(f"Error in LLM call for PMID {pmid}: {str(e)}")
        return {"pmid": pmid, "response": ""}

async def batch_call_leads_with_client(prompts, pmids, model="leads-mistral-7b-v0.3", batch_size=20, endpoint=None, api_key=None, temperature=1.0, max_tokens=1024):
    """Process multiple prompts in parallel batches with proper client management."""
    
    if endpoint is None:
        endpoint = os.getenv("LEADS_ENDPOINT")
    if api_key is None:
        api_key = os.getenv("LEADS_API_KEY")

    async with AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint,
    ) as client:
        all_responses = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]
            batch_pmids = pmids[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            tasks = [call_llm_single(prompt, pmid, client, model, temperature, max_tokens) 
                    for prompt, pmid in zip(batch_prompts, batch_pmids)]
            
            batch_responses = await asyncio.gather(*tasks)
            all_responses.extend(batch_responses)
            
            if i + batch_size < len(prompts):
                await asyncio.sleep(1)
        
        response_dict = {resp["pmid"]: resp["response"] for resp in all_responses}
        ordered_responses = [response_dict[pmid] for pmid in pmids]
        
        return ordered_responses

def call_leads(prompts, prompt_ids=None, model="zifeng-ai/leads-mistral-7b-v1", batch_size=20, endpoint=None, api_key=None, temperature=1.0, max_tokens=1024):
    """Synchronous wrapper for batch processing with LEADS model."""    
    if isinstance(prompts, str):
        prompts = [prompts]
        prompt_ids = [prompt_ids] if prompt_ids else [0]
        single_prompt = True
    else:
        single_prompt = False
        if prompt_ids is None:
            prompt_ids = list(range(len(prompts)))

    # Run the async function
    responses = asyncio.run(
        batch_call_leads_with_client(
            prompts, 
            prompt_ids, 
            model, 
            batch_size, 
            endpoint,
            api_key,
            temperature, 
            max_tokens
        )
    )
    
    return responses[0] if single_prompt else responses
