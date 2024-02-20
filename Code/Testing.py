### TEST BLOCK ###

import os
import re
import requests
from tqdm.notebook import tqdm as tqdm
from tqdm.asyncio import tqdm as async_tqdm

import pandas as pd
import numpy as np

from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import Replicate
from langchain_experimental.chat_models import Llama2Chat
from langchain.schema.output_parser import StrOutputParser

import mercury as mr

import getpass
import os

API_KEY = '' #UMLS REST API KEY

os.chdir('/Users/alis/Library/CloudStorage/OneDrive-Personal/Desktop/_Research/Ongoing_Projects/Submitted/ICD_Code_Paper')

def getCUI_desc(code, system):
    try:
        url = f'https://uts-ws.nlm.nih.gov/rest/content/current/source/{system}/{code}/atoms/preferred?apiKey={API_KEY}'
        response = requests.get(url)
        response.raise_for_status()
        output = response.json()
        umls_cui = output['result']['concept'].split('/')[-1]
        source_desc = output['result']['name']
        return source_desc, code
    except Exception as e:
        return np.nan, code

desc_pattern = re.compile(r'<(.*?)>')
cpt_pattern = re.compile(r'\d+')
icd9cm_pattern = re.compile(r'([\d]{3})[.]?[\d]{0,2}|[A-Z][\d]{2}[.]?[\d]{0,2}')
icd9cm_pattern2 = re.compile(r'\d{4,5}')
icd10cm_pattern = re.compile(r'([A-Z][\d]{2})[.]?[\d]{0,3}[A-Z]?')

    
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

def create_progress_bar(percentage, width=25):
    """Create a text-based progress bar."""
    filled = int(width * percentage)
    bar = '[' + '#' * filled + '-' * (width - filled) + ']'
    return bar

def llm_generate(df_prompt, model_name, temperature, max_tokens, print_output=False):
    codesystem = df_prompt.columns[1].split('_')[0]
    
    if "gpt" in model_name:
        llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    elif "gemini" in model_name:
        google_api_key = os.getenv("GOOGLE_API_KEY") or getpass.getpass("Provide your Google API Key")
        llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=model_name, temperature=temperature, max_tokens=max_tokens) 
    elif "llama" in model_name:
        replicate_api_token = os.getenv("REPLICATE_API_TOKEN") or getpass.getpass("Provide your Replicate API Token")
        llm = Llama2Chat(llm=Replicate(model=model_name, model_kwargs={"temperature":temperature, "max_new_tokens": 30}))
    
    chain = llm | StrOutputParser()
    
    prompt_column = f"{codesystem}_prompt"
    responses = []
    total_prompts = len(df_prompt)
    
    for index, prompt in enumerate(df_prompt[prompt_column]):
        if print_output == False:
            progress_percentage = (index) / total_prompts
            progress_bar = create_progress_bar(progress_percentage)
            print(f'\rLLM running: {progress_bar} {index}/{total_prompts}', end='')
        
        response = extractcode(chain.invoke(prompt),codesystem)
        responses.append(response)
        
        if print_output == False:
            progress_percentage = (index + 1) / total_prompts
            progress_bar = create_progress_bar(progress_percentage)
            print(f'\rLLM running: {progress_bar} {index + 1}/{total_prompts}', end='')
        
        if print_output:
            print(response)
    if print_output == False:
        print()
    
    return responses
