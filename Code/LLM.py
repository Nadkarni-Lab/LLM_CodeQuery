import os
import asyncio
from aiolimiter import AsyncLimiter
import re
import pickle
from tqdm.notebook import tqdm as tqdm
from tqdm.asyncio import tqdm as async_tqdm

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import Replicate
from langchain_experimental.chat_models import Llama2Chat
from langchain.schema.output_parser import StrOutputParser

import getpass

API_KEY = '' #UMLS REST API KEY

os.chdir('/Users/alis/Library/CloudStorage/OneDrive-Personal/Desktop/_Research/Ongoing_Projects/Submitted/ICD_Code_Paper')

desc_pattern = re.compile(r'<(.*?)>')
cpt_pattern = re.compile(r'\d+')
icd9cm_pattern = re.compile(r'([\d]{3})[.]?[\d]{0,2}|[A-Z][\d]{2}[.]?[\d]{0,2}')
icd9cm_pattern2 = re.compile(r'\d{4,5}')
icd10cm_pattern = re.compile(r'([A-Z][\d]{2})[.]?[\d]{0,3}[A-Z]?')

## LLM Functions ##
def extractdesc(input_string):
    pattern = desc_pattern
    match = re.search(pattern, str(input_string))
    if match:
        return match.group(1)
    else:
        return None
    
def extractcode(input_string, codesystem):
    pattern = None
    if codesystem == "CPT":
        pattern = cpt_pattern
    elif codesystem == "ICD9CM":
        pattern = icd9cm_pattern
        
        # Check if it matches icd9cm_pattern2
        match2 = icd9cm_pattern2.search(str(input_string))
        if match2:
            matched_string = match2.group()
            
            # Add a '.' after the 3rd digit if the matched string's length is more than 3
            if len(matched_string) > 3:
                return matched_string[:3] + '.' + matched_string[3:]
        
    elif codesystem == "ICD10CM":
        pattern = icd10cm_pattern

    if pattern:
        match = pattern.search(str(input_string))
        if match:
            return match.group()
    else:
        return None
    
def load_cache(filename):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return {}

def save_cache(data, filename):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

async def async_invoke(chain, prompt_text, index, limiter, semaphore, api_timeout=10, max_attempts=3):
    attempt = 0
    while attempt < max_attempts:
        try:
            async with limiter:
                async with semaphore:  
                    await asyncio.sleep(0.05)
                    response = await asyncio.wait_for(chain.ainvoke(prompt_text), api_timeout)
                    return response
        except asyncio.TimeoutError:
            print(f"Timed out description: {index}  {extractdesc(prompt_text)}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            attempt += 1

async def generate_responses_concurrently(model_name, temperature, max_tokens, code_datasets, codesystem, batch_size, raw_path, RESULTS_CACHE):
    model_rps_limits = {
        "gpt": 160,
        "gemini": 1,
        "llama": 10
    }
    
    model_concurrency_limits = {
        "gpt": 400,  
        "gemini": 10,  
        "llama": 8
    }

    if "gpt" in model_name:
        llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        model_fam="gpt"
    elif "gemini" in model_name:
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
        llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model=model_name, temperature=temperature, max_tokens=max_tokens) 
        model_fam="gemini"  
    elif "llama" in model_name:
        if "REPLICATE_API_TOKEN" not in os.environ:
            os.environ["REPLICATE_API_TOKEN"] = getpass.getpass("Provide your Replicate API Token")
        llm = Llama2Chat(llm=Replicate(model=model_name, model_kwargs={"temperature":temperature, "max_new_tokens": 30}))
        model_fam="llama"
    
    chain = llm | StrOutputParser()
    semaphore = asyncio.Semaphore(model_concurrency_limits[model_fam])
    limiter = AsyncLimiter(model_rps_limits[model_fam], 1)
    
    # Determine the starting batch index based on saved progress
    model_name_clean = model_name.replace(".", "").split(":")[0]
    start_index = len(RESULTS_CACHE.get((codesystem, model_name_clean), []))

    total = len(code_datasets[codesystem][f"{codesystem}_prompt"])
    for i in async_tqdm(range(start_index, total, batch_size), desc="Processing batches"):
        batch_start = i
        batch_end = min(i + batch_size, total)
        batch = code_datasets[codesystem][f"{codesystem}_prompt"][batch_start:batch_end]
        tasks = [async_invoke(chain, prompt, i, semaphore, limiter) for i, prompt in enumerate(batch, start=batch_start)]
        responses_batch = await asyncio.gather(*tasks)

        if (codesystem, model_name_clean) in RESULTS_CACHE:
            RESULTS_CACHE[(codesystem, model_name_clean)].extend(responses_batch)
        else:
            RESULTS_CACHE[(codesystem, model_name_clean)] = responses_batch
        save_cache(RESULTS_CACHE, raw_path)
        
        #print(i, responses_batch)

    return RESULTS_CACHE[(codesystem, model_name_clean)]


async def run_llms(model_list, code_datasets, suffix):
    codesystem_list = code_datasets.keys()
    temp = 0.2
    raw_path = f"Output/Intermediate/results{suffix}.pkl"
    clean_path = f"Output/Intermediate/results{suffix}_clean.pkl"
    
    RESULTS_CACHE = load_cache(raw_path)
    RESULTS_CACHE_CLEAN = {}

    for codesystem in codesystem_list:
        for model_name in model_list:
            model_name_clean = model_name.replace(".", "").split(":")[0]
            
            print("\n", codesystem, model_name_clean)

            #RESULTS_CACHE[(codesystem, model_name_clean)] = [] # Uncomment if you want to overrwite results

            retry_attempts = 3
            while retry_attempts > 0:
                try:
                    responses = await generate_responses_concurrently(model_name=model_name, 
                                                                      temperature=temp, 
                                                                      max_tokens=50, 
                                                                      code_datasets=code_datasets, 
                                                                      codesystem=codesystem,
                                                                      batch_size=100,
                                                                      raw_path=raw_path,
                                                                      RESULTS_CACHE=RESULTS_CACHE
                                                                      )
                    break
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    retry_attempts -= 1
                    if retry_attempts == 0:
                        print(f"Retry attempts exhausted for {codesystem} {model_name}")

            for key in RESULTS_CACHE.keys():
                code_system = key[0]
                extracted_codes = pd.Series(RESULTS_CACHE[key]).apply(lambda r: extractcode(r, code_system)).str.rstrip('.')
                RESULTS_CACHE_CLEAN[key] = extracted_codes.tolist()
            save_cache(RESULTS_CACHE_CLEAN, clean_path)
            print("Clean cache saved")

    save_cache(RESULTS_CACHE, raw_path)
    for key in RESULTS_CACHE.keys():
        code_system = key[0]
        extracted_codes = pd.Series(RESULTS_CACHE[key]).apply(lambda r: extractcode(r, code_system)).str.rstrip('.')
        RESULTS_CACHE_CLEAN[key] = extracted_codes.tolist()
    save_cache(RESULTS_CACHE_CLEAN, clean_path)    
    print("Clean cache saved")
