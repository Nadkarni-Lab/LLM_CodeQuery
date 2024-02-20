import os
import aiohttp
import asyncio
from aiolimiter import AsyncLimiter
import pickle
import string
from tqdm.notebook import tqdm as tqdm
from tqdm.asyncio import tqdm as async_tqdm

import pandas as pd
import numpy as np

import os

API_KEY = '' #UMLS REST API KEY

os.chdir('/Users/alis/Library/CloudStorage/OneDrive-Personal/Desktop/_Research/Ongoing_Projects/Submitted/ICD_Code_Paper')

limiter = AsyncLimiter(20, 1)  # 20 calls per second

def load_umls_cache(filename="Output/Intermediate/UMLS_CACHE.pkl"):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return {}

UMLS_CACHE = load_umls_cache()

async def getCUI_desc(code, system, session):
    
    if (code, system) in UMLS_CACHE:
        result = UMLS_CACHE.get((code, system))
        if pd.notna(result[1]):
            umls_cui = result[0]
            source_desc = result[1] 
            return source_desc, umls_cui, None, code
        # else:
        #     return np.nan, np.nan, None, code

    async with limiter:  
        try:
            url = f'https://uts-ws.nlm.nih.gov/rest/content/current/source/{system}/{code}/atoms/preferred?apiKey={API_KEY}'
            async with session.get(url) as response:
                response.raise_for_status()
                output = await response.json()
                umls_cui = output['result']['concept'].split('/')[-1]
                source_desc = output['result']['name']
                UMLS_CACHE[(code, system)] = (umls_cui, source_desc)
                return source_desc, umls_cui, None, code
        except Exception as e:
            UMLS_CACHE[(code, system)] = (np.nan, np.nan)
            return np.nan, np.nan, e, code

def save_desc_cache():
    with open('Output/Intermediate/UMLS_CACHE.pkl', 'wb') as pickle_file:
        pickle.dump(UMLS_CACHE, pickle_file)
    print("Cache saved")

## PROMPT TEMPLATE CREATION ##
async def prompt_dataset_with_langchain(file_path):
    session = None
    try:
        session = aiohttp.ClientSession()
        df = pd.read_excel(file_path, engine='openpyxl', index_col=0)
        basefilename = os.path.splitext(os.path.basename(file_path))[0]
        codesystem = basefilename.upper().translate(str.maketrans('', '', string.punctuation))
            
        code_col = {"ICD9CM":"DiagnosisValue","ICD10CM":"DiagnosisValue","CPT":"CptCode"}
        desc_col = {"ICD9CM":"DisplayString","ICD10CM":"DisplayString","CPT":"Name"}
            
        tasks = {df.index[i]: getCUI_desc(row[code_col[codesystem]], codesystem, session) for i, (index, row) in enumerate(df.iterrows())}
        progress_bar = async_tqdm(total=len(tasks), desc="Processing API Calls")

        error_messages = []
        code_description_pairs = []

        for index in df.index:
            desc, cui, err, code = await tasks[index]
            if err:
                error_messages.append(f"\tNo result, error {err}: {code} {desc}")
            else:
                code_description_pairs.append((index, code, desc))
            progress_bar.update(1)

        progress_bar.close()
        
        code_description_pairs.sort(key=lambda x: x[0])
    
        examples = {"ICD9CM":"045.10","ICD10CM":"M24.131","CPT":"84120"}
            
        # Creating DataFrame
        data = [{
            f"{codesystem}_code": code,
            f"{codesystem}_EHRdesc": df.at[index, desc_col[codesystem]],
            f"{codesystem}_codedesc": desc,
            f"{codesystem}_prompt": f"What is the most correct {codesystem} billing code for this description: <{desc}>. \
                \nOnly generate a single, VALID {codesystem} billing code. Do not explain. ALWAYS respond in the following format: \
                \nCode: {examples.get(codesystem)}",
            f"{codesystem}_count": df.at[index, "count"]
        } for index, code, desc in code_description_pairs if desc is not None]

        df = pd.DataFrame(data)

        # Print errors after progress bar setup
        for error_message in error_messages:
            print(error_message)
            
        if codesystem == "CPT":
            regex_pattern = r'^\d{5}$'
            df = df[df['CPT_code'].str.match(regex_pattern, na=False)]
        df = df.dropna(subset=[f'{codesystem}_codedesc'])
        
        df.reset_index(drop=True, inplace=True)
            
        return df
        
    except asyncio.CancelledError:
        # Handle cancellation outside task execution
        print("Operation cancelled by user. Cleaning up...")
    finally:
        save_desc_cache()
        if session and not session.closed:
            await session.close()
            
def sample_optimized(df, num_samples, random_state, df2=None):
    df = df.copy()
    
    system = df.columns[1].split('_')[0]
    count_col = f'{system}_count'
    code_col = f'{system}_code'

    if df2 is not None:
        df = df[~df[code_col].isin(df2[code_col])]

    # Convert counts to probabilities
    df['probability'] = df[count_col] / df[count_col].sum()

    # Sample rows based on the probability, ensuring uniqueness
    sample_df = df.drop_duplicates(subset=[code_col]).sample(n=num_samples, weights='probability', replace=False, random_state=random_state)
    sample_df.reset_index(drop=True,inplace=True)

    return sample_df

async def process_gen_codes(df, codesystem, results_dict):
    async with aiohttp.ClientSession() as session:
        # Create tasks for the initial codes
        initial_tasks = [asyncio.ensure_future(getCUI_desc(code, codesystem, session)) 
                         for code in df[f'{codesystem}_code']]
        
        # Wait for all initial tasks to complete
        initial_responses = await async_tqdm.gather(*initial_tasks, desc=f"Processing {codesystem} original")
        
        # Process the initial responses and populate the dataframe
        for idx, response in zip(df.index, initial_responses):
            source_desc, umls_cui, _, _ = response
            df.at[idx, f'{codesystem}_CUI'] = umls_cui
        
        # Now handle the additional getCUI_desc calls for different models
        model_list = ["gpt-35-turbo-0301",
              "gpt-35-turbo-0613",
              "gpt-35-turbo-1106",
              "gpt-4-0314",
              "gpt-4-0613",
              "gpt-4-1106-preview",
              "gemini-pro",
              "meta/llama-2-70b-chat",
              ]
        
        for model_name in model_list:
            model_name_clean = model_name.replace(".", "").split(":")[0]
            df[model_name_clean] = results_dict[(codesystem, model_name_clean)]
            df[f'{model_name_clean}_CUI'] = pd.Series(dtype='object')
            df[f'{model_name_clean}_desc'] = pd.Series(dtype='object')
            
            model_tasks = [asyncio.ensure_future(getCUI_desc(code, codesystem, session)) 
                           for code in df[model_name_clean]]

            model_responses = await async_tqdm.gather(*model_tasks, desc=f"Processing {codesystem} {model_name}")

            for idx, response in zip(df.index, model_responses):
                source_desc, umls_cui, _, _ = response
                df.at[idx, f'{model_name_clean}_CUI'] = umls_cui
                df.at[idx, f'{model_name_clean}_desc'] = source_desc
        
        # Save the cache after processing
        save_desc_cache()

    return df

def semantic_pairs_manual(excel_filename, suffix):
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        man_datasets = {}
        
        for codesystem in ['ICD9CM', 'ICD10CM', 'CPT']:
            df = pd.read_parquet(f"Output/Intermediate/{codesystem}_parsed{suffix}.parquet")
            df_new = pd.DataFrame()
            pairs_list = []

            model_list = [
                "gpt-35-turbo-0301",
                "gpt-35-turbo-0613",
                "gpt-35-turbo-1106",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-1106-preview",
                "gemini-pro",
                "meta/llama-2-70b-chat",
            ]
            
            for model_name in model_list:
                model_name_clean = model_name.replace(".", "")
                pairs_list.extend(zip(df[f'{codesystem}_codedesc'], df[f"{model_name_clean}_desc"]))

            # Create a new DataFrame with 'codedesc' and 'desc' columns
            df_new[['codedesc', 'desc']] = pd.DataFrame(pairs_list, columns=['codedesc', 'desc'])

            df_new.drop_duplicates(inplace=True)
            df_new.dropna(inplace=True)
            df_new = df_new[df_new['codedesc'] != df_new['desc']]

            df_new.to_excel(writer, sheet_name=codesystem, index=False)

            # Add df_new to man_datasets
            man_datasets[codesystem] = df_new
            
### CREATE BILLABLE ICD CODE DICTIONARIES ###

## ICD9 import ##
df_icd9 = pd.read_excel('Raw/CMS32_DESC_LONG_SHORT_DX.xlsx', engine='openpyxl', usecols=["DIAGNOSIS CODE","LONG DESCRIPTION"], converters={'DIAGNOSIS CODE':str,'LONG DESCRIPTION':str})

## ICD 10 CM import ##
df_icd10cm = pd.read_fwf('Raw/icd10cm_codes_2023.txt', colspecs=[(0,7),(8,400)], header=None, converters={0:str, 1: str})
df_icd10cm_addendum = pd.read_fwf('Raw/icd10cm_codes_addenda_2023.txt', colspecs='infer', infer_nrows=100, header=None, type=str)

# add addenda
mask_add = df_icd10cm_addendum[0] == "Add:"
added_data = df_icd10cm_addendum.loc[mask_add, [1, 2]]
added_data = added_data.rename(columns={1:0, 2:1})
df_icd10cm = pd.concat([df_icd10cm, added_data], ignore_index=True)

mask_del = df_icd10cm_addendum[0] == "Delete:"
delete_values = df_icd10cm_addendum.loc[mask_del, 1]
df_icd10cm = df_icd10cm.loc[~df_icd10cm[0].isin(delete_values)]

mask_rev = df_icd10cm_addendum.iloc[:, 0] == "Revise to:"

for _, row in df_icd10cm_addendum[mask_rev].iterrows():
    mask_rev2 = df_icd10cm.iloc[:, 0] == row.iloc[1]
    df_icd10cm.loc[mask_rev2, df_icd10cm.columns[1]] = row.iloc[2]

# rename columns for dataframes import from text files
df_icd10cm.rename(columns={0:"DIAGNOSIS CODE", 1:"LONG DESCRIPTION"}, inplace=True)

# set dict
ICD_9_dict = dict(zip(df_icd9["DIAGNOSIS CODE"], df_icd9["LONG DESCRIPTION"]))
ICD_10_CM_dict = dict(zip(df_icd10cm["DIAGNOSIS CODE"], df_icd10cm["LONG DESCRIPTION"]))