import os, json, re, pickle, subprocess
from tqdm.notebook import tqdm as tqdm
from tqdm import tqdm as tqdm_conc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

import mercury as mr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import scipy.stats
from scipy.stats import chi2
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score
from bert_score import score
from icdcodex import icd2vec, hierarchy
import pandas as pd
from nltk.translate import meteor
from nltk import word_tokenize
#import nltk
#nltk.download('punkt')

API_KEY = '' #UMLS REST API KEY

os.chdir('/Users/alis/Library/CloudStorage/OneDrive-Personal/Desktop/_Research/Ongoing_Projects/Submitted/ICD_Code_Paper')

def code_histogram():
    fontsize=10
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), dpi=300)

    for i, (codesystem, ax) in enumerate(zip(['ICD9CM', 'ICD10CM', 'CPT'], axes.flatten())):
        df = pd.read_parquet(f"Output/Intermediate/{codesystem}prompts.parquet")
        
        if codesystem == 'ICD9CM':
            codesystem_title = 'ICD-9-CM'
        elif codesystem == 'ICD10CM':
            codesystem_title = 'ICD-10-CM'
        else:
            codesystem_title = codesystem
        
        df = df[[f'{codesystem}_code', f'{codesystem}_codedesc', f'{codesystem}_count']]
        bins = np.logspace(0, 6.25, num=28)
        
        # Plotting in the respective subplot
        ax.hist(df[f'{codesystem}_count'], bins=bins, edgecolor='black')
        ax.set_xscale('log')  
        ax.set_xlim(1,1000000)
        
        # Set labels and title with specific font size
        ax.set_xlabel('Frequency of each code', fontsize=fontsize)
        ax.set_ylabel('Number of codes in bin', fontsize=fontsize)
        ax.set_title(f'{codesystem_title}', fontsize=fontsize)

        # Set the font size for tick labels
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(fontsize)
            
    plt.tight_layout()
    plt.savefig(f"Output/code_histogram.png", format='png', dpi=300)

    # Show the plot
    plt.show()

def convert_to_nested_json(d):
    nested_json = {}
    for key, value in d.items():
        if isinstance(key, tuple):
            temp = nested_json
            for item in key[:-1]:  # Iterate over the tuple except for the last element
                temp = temp.setdefault(item, {})  # Create nested dictionaries
            temp[key[-1]] = value  # Set the value for the last element
        else:
            nested_json[key] = value
    return nested_json

def displayJSONpretty(json_file):
    mr.JSON(json_file)

def getcolpercent(df_result, column, model, analysis, metric):
    analysis[metric][model] = str(round(df_result[column].mean()*100,2)) + "%"

def codedigit_check(code, gen_code):
    match_count = 0
    total_count = 0
    match_list = []
    
    for i, orig_char in enumerate(code):
        if orig_char != ".":
            total_count += 1 
            if isinstance(gen_code, str) == True:
                if i < len(gen_code):
                    gen_char = gen_code[i]
                    if orig_char == gen_char :
                        match_count += 1
                        match_list.append(total_count)
    return match_count/total_count, match_list, total_count

def get_billable_code_dict():
    ### ICD9 import ###
    df_icd9 = pd.read_excel('Raw/CMS32_DESC_LONG_SHORT_DX.xlsx', engine='openpyxl', usecols=["DIAGNOSIS CODE","LONG DESCRIPTION"], converters={'DIAGNOSIS CODE':str,'LONG DESCRIPTION':str})

    ### ICD 10 CM import ###
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

    ### CPT import ###
    
    billable_dict ={}
    billable_dict['ICD9CM'] = dict(zip(df_icd9["DIAGNOSIS CODE"], df_icd9["LONG DESCRIPTION"]))
    billable_dict['ICD10CM'] = dict(zip(df_icd10cm["DIAGNOSIS CODE"], df_icd10cm["LONG DESCRIPTION"]))
    # billable_dict['CPT'] = 
    return billable_dict

def quickanalyze(model_list, file_path, suffix=""):
    analysis_dict={}
    df_results_dict={}
    billable_dict = get_billable_code_dict()
    
    with open(file_path, "rb") as file:
        results_dict = pickle.load(file)
    
    for codesystem in ["ICD9CM", "ICD10CM", "CPT"]:
        analysis = {'ExactMatch': {}, 'BillableCode':{}, 'LengthMatch':{}, 'DigitMatch_Ovr': {}, 'DigitMatch_Ind':{}}#, 'Part_Code': {}}
        df_result = pd.read_parquet(f"Output/Intermediate/{codesystem}prompts{suffix}.parquet")
        df_result = df_result.fillna("")

        for model in model_list:
            model_name_clean = model.replace(".", "").split(":")[0]
            df_result[model_name_clean] = results_dict[(codesystem, model_name_clean)]

            fmatch = model_name_clean + "_ExactMatch"
            df_result[fmatch] = (df_result[model_name_clean] == df_result[f"{codesystem}_code"]).astype(int)
            getcolpercent(df_result, fmatch, model_name_clean, analysis, "ExactMatch")
            
            if codesystem != "CPT":
                bmatch = model_name_clean + "_BillableCode"
                df_result[bmatch] = df_result[model_name_clean].apply(lambda x: 1 if x is not None and x.replace(".", "") in billable_dict[codesystem].keys() else 0)
                getcolpercent(df_result, bmatch, model_name_clean, analysis, "BillableCode")
            
            lmatch = model_name_clean + "_LengthMatch"
            df_result[lmatch] = (df_result[model_name_clean].str.len() == df_result[f"{codesystem}_code"].str.len()).astype(int)
            getcolpercent(df_result, lmatch, model_name_clean, analysis, "LengthMatch")

            dmatch = model_name_clean + "_DigitMatch_Ovr"
            df_result[dmatch], df_result[f'{dmatch}_list'], df_result['code_length'] = zip(*df_result.apply(lambda row: codedigit_check(row[f"{codesystem}_code"], row[model_name_clean]), axis=1))
            getcolpercent(df_result, dmatch, model_name_clean, analysis, "DigitMatch_Ovr")
            
            digit_match = df_result[f'{dmatch}_list'].explode()
            analysis["DigitMatch_Ind"][model_name_clean] = {
                key: f"{round(value / (df_result['code_length'] >= key).sum() * 100, 2):.2f}%"
                if (df_result['code_length'] >= key).sum() !=0
                else '0.00%'
                for key, value in digit_match.value_counts().items()
            }
            
            #pmatch = model_name_clean+"_Part_Code"
            #df_result[pmatch] = (df_result[model_name_clean].str.split('.').str[0] == df_result[f"{codesystem}_code"].str.split('.').str[0]).astype(int)
            #getcolpercent(df_result, pmatch, model_name_clean, analysis, "Part_Code")
        
        df_results_dict[codesystem] = df_result
        analysis_dict[codesystem] = analysis
        
    with open(f"Output/Intermediate/analysis{suffix}.json", "w") as file:
        json.dump(analysis_dict, file)
    
    with open(f"Output/Intermediate/df_analysis{suffix}.pkl", "wb") as file:
        pickle.dump(df_results_dict, file)

    return analysis_dict, df_results_dict
    
global_embeddings = None
global global_dict
global_dict = {} 

def load_embeddings(file_path):
    global global_embeddings
    df = pd.read_parquet(file_path)
    df.set_index(df.columns[0], inplace=True)
    
    # Convert the DataFrame to a SciPy sparse matrix
    sparse_matrix = csr_matrix(df.values)
    
    # Create a Series to map IDs to row indices
    id_to_row = {id_val: idx for idx, id_val in enumerate(df.index)}
    global_embeddings = sparse_matrix, id_to_row 

    return global_embeddings

def create_icdcodexvectors(overwrite=False, suffix=""):
    global global_dict
    file_path = f'Output/Intermediate/icdcodex_dict{suffix}.pkl'
    
    model_list = ["gpt-3.5-turbo-0301",
              "gpt-3.5-turbo-0613",
              "gpt-3.5-turbo-1106",
              "gpt-4-0314",
              "gpt-4-0613",
              "gpt-4-1106-preview",
              "gemini-pro",
              "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
              ]
    
    #Check if the file exists
    if os.path.isfile(file_path) and overwrite == False:
        with open(file_path, 'rb') as file:
            global_dict = pickle.load(file)
    else:
        for codesystem in ["ICD9CM", "ICD10CM"]:
            df = pd.read_parquet(f'Output/Intermediate/{codesystem}_parsed{suffix}.parquet')
            
            model_names_clean = [name.replace(".", "").split(":")[0] for name in model_list]
            id_cols = [f"{codesystem}_code"] + model_names_clean
            embedder = icd2vec.Icd2Vec(workers=-1)

            if codesystem == "ICD9CM":
                code_list = pd.concat([df[col].astype(str).str.replace(".", "") for col in id_cols]).unique()
                embedder.fit(*hierarchy.icd9())
            elif codesystem == "ICD10CM":
                code_list = pd.concat([df[col].astype(str) for col in id_cols]).unique()
                embedder.fit(*hierarchy.icd10cm(version="2023"))

            global_dict[codesystem] = {}
            
            for code in tqdm_conc(code_list):
                try:
                    vector = embedder.to_vec([code])
                    global_dict[codesystem][code] = vector[0].flatten()
                except KeyError:
                    global_dict[codesystem][code] = None

        # Save global_dict to a file
        with open(file_path, 'wb') as file:
            pickle.dump(global_dict, file)

    return global_dict

def get_vector(embeddings, id_val):
    sparse_matrix, id_to_row = embeddings
    row_index = id_to_row.get(id_val)
    return sparse_matrix[row_index].toarray().flatten() if row_index is not None else None

@lru_cache(maxsize=None)  # Caching for repeated vector pair calculations
def calc_cosine_sim(id_pairs, metric, codesystem):
    id1, id2 = id_pairs
    vector1, vector2 = None, None
    
    if metric == "icdcodex" and codesystem != "CPT":
        if codesystem == "ICD9CM":
            id1 = id1.replace(".", "")
            id2 = id2.replace(".", "")

        vector1 = global_dict[codesystem].get(id1)
        vector2 = global_dict[codesystem].get(id2)
        
    elif metric == "cui2vec" and codesystem != "CPT":
        vector1 = get_vector(global_embeddings, id1)
        vector2 = get_vector(global_embeddings, id2)

    if vector1 is not None and vector2 is not None:
        cosine_distance = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
        if metric == "icdcodex":
            cosine_distance = (cosine_distance + 1) / 2  # Normalizing to the range of 0 to 1
        similarity = round(cosine_distance, 3)
    else:
        similarity = pd.NA
    
    return similarity

@lru_cache(maxsize=None)  # Caching for repeated vector pair calculations
def calc_meteor_score(pair):
    desc1, desc2 = pair
    score = round(meteor([word_tokenize(desc1)], word_tokenize(desc2)), 3)
    return score

def calc_score(result_dict, df, codesystem, model_name, score_map, metric):
    model_name_clean = model_name.replace(".", "").split(":")[0]
    df_result = pd.DataFrame()
    
    if metric == "icdcodex":
        vector1_col = f"{codesystem}_code"
        vector2_col = f"{model_name_clean}{score_map[metric]}"
    elif metric == "bertscore" or metric == "meteor":
        vector1_col = f"{codesystem}_codedesc"
        vector2_col = f"{model_name_clean}{score_map[metric]}"
    elif metric=="cui2vec": 
        vector1_col = f"{codesystem}{score_map[metric]}"
        vector2_col = f"{model_name_clean}{score_map[metric]}"
            
    # Filter out rows where either code is None
    valid_pairs = df[[vector1_col, vector2_col]].dropna()
    
    # Pre-calculate similarities for unique pairs
    unique_pairs = valid_pairs.drop_duplicates()
    precalculated_results = {}
    
    if metric == "bertscore":
        refs = unique_pairs[vector1_col].astype(str).tolist()
        cands = unique_pairs[vector2_col].astype(str).tolist()

        if cands and refs:
            P, R, F1 = score(cands, refs, lang='en-sci', verbose=False)
            F1_values = F1.numpy()
            
            for i, pair in enumerate(unique_pairs.itertuples(index=False, name=None)):
                precalculated_results[pair] = F1_values[i]
    
    elif metric == "meteor":
        precalculated_results = {
            pair: calc_meteor_score(pair) 
            for pair in unique_pairs.itertuples(index=False, name=None)
            }

    else:
        precalculated_results = {
            pair: calc_cosine_sim(pair, metric, codesystem) 
            for pair in unique_pairs.itertuples(index=False, name=None)
            }
    
    # Apply precalculated results
    df_result[f'{model_name_clean}_{metric}'] = df.apply(
        lambda row: precalculated_results.get((row[vector1_col], row[vector2_col])), axis=1
    )
    
    df_result[f'{model_name_clean}_{metric}'] = pd.to_numeric(df_result[f'{model_name_clean}_{metric}'], errors='coerce')
    df_result[f'{model_name_clean}_{metric}'] = df_result[f'{model_name_clean}_{metric}'].apply(lambda x: round(x, 3) if pd.notna(x) else x)
    df_result[f'{model_name_clean}_{metric}'] = df_result[f'{model_name_clean}_{metric}'].round(3)
    
    result_dict[codesystem][f"{metric} vectors"][model_name_clean] = int(df_result[f'{model_name_clean}_{metric}'].count())
                
    mean_value = float(df_result[f'{model_name_clean}_{metric}'].mean())
    result_dict[codesystem][metric][model_name_clean] = round(mean_value,3)
    
    return df_result

def get_score(result_dict, datasets, model_list, metric, suffix=""):
    print(metric)
    
    score_map = {"cui2vec":"_CUI", "bertscore":"_desc", "icdcodex":"", "meteor":"_desc"}

    for codesystem in ["ICD9CM", "ICD10CM", "CPT"]:
        df = pd.read_parquet(f'Output/Intermediate/{codesystem}_parsed{suffix}.parquet')
        
        if codesystem not in datasets:        
            datasets[codesystem] = df.copy()
        
        if codesystem not in result_dict:
            result_dict[codesystem] = {}         
               
        if metric not in result_dict[codesystem]:   
            result_dict[codesystem][metric] = {}
            result_dict[codesystem][f"{metric} vectors"] = {}

            with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
                future_to_model = {executor.submit(calc_score, result_dict, df, codesystem, model_name, score_map, metric): model_name for model_name in model_list}

                for future in tqdm_conc(as_completed(future_to_model), total=len(future_to_model), desc=f"Processing {codesystem}"):
                    df_result = future.result()
                    model_name_clean = future_to_model[future].replace(".", "").split(":")[0]
                    datasets[codesystem][f'{model_name_clean}_{metric}'] = df_result[f'{model_name_clean}_{metric}']
                    datasets[codesystem].loc[(datasets[codesystem][f'{model_name_clean}_desc'] == datasets[codesystem][f'{codesystem}_codedesc']), f'{model_name_clean}_{metric}'] = 1

    with open(f"Output/Intermediate/analysis{suffix}_automatedscores.json", "w") as file:
        json.dump(result_dict, file)
        
    with open(f"Output/Intermediate/df_analysis{suffix}_automatedscores.pkl", "wb") as file:
        pickle.dump(datasets, file)

    return result_dict, datasets

def run_meteor(java_command):
    try:
        result = subprocess.run(java_command, capture_output=True, text=True)
        return result
    except Exception as e:
        print(f"Error running subprocess: {e}")

def process_model(df, codesystem, model_name):
    model_name_clean = model_name.replace(".", "").split(":")[0]
    model_name_clean_filepath = model_name.replace(".", "").split("/")[0]
    
    cand_col = f'{model_name_clean}_desc'
    ref_col = f'{codesystem}_codedesc'   

    meteor_csv_path = f"{os.getcwd()}/Output/Intermediate/meteor"    
    meteor_jar_path = f"{os.getcwd()}/Raw/meteor-1.5/meteor-1.5.jar"
    model_filepath = f'{meteor_csv_path}/{codesystem}_{model_name_clean_filepath}.txt'
    ref_filepath = f'{meteor_csv_path}/{codesystem}_{model_name_clean_filepath}_ref.txt'
    
    valid_pairs = df[[ref_col, cand_col]].dropna().drop_duplicates()
    valid_pairs.reset_index(drop=True, inplace=True)
    
    valid_pairs[ref_col].to_csv(ref_filepath, index=False, encoding='utf-8')
    valid_pairs[cand_col].to_csv(model_filepath, index=False, encoding='utf-8')
    
    java_command = ["java", "-Xmx1G", "-jar", meteor_jar_path, 
                    model_filepath, ref_filepath, 
                    "-l", "en", "-norm"]

    result = run_meteor(java_command)
        
    return model_name_clean, result, valid_pairs

def meteor_15(result_dict, datasets, model_list, suffix):
    print("Running meteor15")
    for codesystem in ["ICD9CM", "ICD10CM", "CPT"]:
        
        ref_col = f'{codesystem}_codedesc'
        
        df = pd.read_parquet(f'Output/Intermediate/{codesystem}_parsed{suffix}.parquet')
        
        if codesystem not in datasets:        
            datasets[codesystem] = df

        if codesystem not in result_dict:
            result_dict[codesystem] = {}
        if "meteor15" not in result_dict[codesystem]:
            result_dict[codesystem]["meteor15"] = {}
            result_dict[codesystem][f"meteor15 vectors"] = {}
        
        df_result = pd.DataFrame()
        df_result[f'{codesystem}_codedesc'] = datasets[codesystem][f'{codesystem}_codedesc'].copy()
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            futures = {executor.submit(process_model, df, codesystem, model_name): model_name for model_name in model_list}
            
            for future in tqdm_conc(as_completed(futures), total=len(futures), desc=f"Processing {codesystem}"):
                model_name_clean, result, pairs = future.result()
                
                metric_col = f'{model_name_clean}_meteor15'
                cand_col =f'{model_name_clean}_desc'
                
                if result.returncode == 0:
                    output = result.stdout
                    
                    # Regular expression to find segment scores
                    pattern = r"Segment (\d+) score:\t([0-9.]+)"
                    matches = re.findall(pattern, output)
                    segment_scores = {int(segment)-2: round(float(score),3) for segment, score in matches}
                    
                    pairs[metric_col] = pd.Series(segment_scores)
                    
                    count = int(pairs[metric_col].count())
                    mean_value = float(pairs[metric_col].mean())
                    
                    result_dict[codesystem][f"meteor15 vectors"][model_name_clean] = count
                    result_dict[codesystem]["meteor15"][model_name_clean] = round(mean_value,3)
                    
                    metric_dict = pairs.groupby([ref_col, cand_col])[metric_col].apply(lambda x: x.iloc[0]).to_dict()
                    
                    datasets[codesystem][metric_col] = datasets[codesystem].apply(lambda row: metric_dict.get((row[ref_col], row[cand_col]), None), axis=1)
                    datasets[codesystem].loc[(datasets[codesystem][f'{model_name_clean}_desc'] == datasets[codesystem][f'{codesystem}_codedesc']), metric_col] = 1
                    
        with open(f"Output/Intermediate/analysis{suffix}_automatedscores.json", "w") as file:
            json.dump(result_dict, file)
            
        with open(f"Output/Intermediate/df_analysis{suffix}_automatedscores.pkl", "wb") as file:
            pickle.dump(datasets, file)

    return result_dict, datasets

def frequencychart(model_list, suffix=""):
    with open(f"Output/Intermediate/df_analysis{suffix}.pkl", "rb") as file:
        df_results_dict = pickle.load(file)

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14})

    # Prepare an empty DataFrame for the aggregated data
    aggregated_data = pd.DataFrame()

    for model_name in model_list:
        model_name_clean = model_name.replace(".", "").split(":")[0]
        fmatch_col_name = model_name_clean + "_Full_Code"

        for codesystem in ["ICD9CM", "ICD10CM", "CPT"]:
            df_result = df_results_dict[codesystem].fillna("")
            df_result[fmatch_col_name] = (df_result[model_name_clean] == df_result[f"{codesystem}_code"]).astype(int)

            bins = np.logspace(0, 7, num=8)

            # Assign bins to the 'count' column

            bin_categories = ["<10$^1$", "10$^1$-10$^2$", "10$^2$-10$^3$", "10$^3$-10$^4$", "10$^4$-10$^5$", ">10$^5$", ""]
            
            df_result['bin'] = pd.cut(df_result[f'{codesystem}_count'], bins=bins, labels=bin_categories, right=False, include_lowest=True)
            df_result['bin'] = pd.Categorical(df_result['bin'], categories=bin_categories)

            # Calculate mean and count for each bin
            aggregated_stats = df_result.groupby('bin', observed=False)[fmatch_col_name].agg(['mean', 'count', 'sem']).reset_index(drop=True)
            aggregated_stats['code system'] = codesystem
            aggregated_stats['model'] = model_name_clean
            aggregated_stats['bin'] = df_result['bin'].cat.categories
            
            # Append to the aggregated data
            aggregated_data = pd.concat([aggregated_data, aggregated_stats], ignore_index=True)

    # Plotting - one subplot per model
    n_models = len(model_list)
    n_cols = 2  # for example, 2 columns
    n_rows = n_models // n_cols + (n_models % n_cols > 0)

    plt.figure(figsize=(12 * n_cols, 6 * n_rows))  # Adjust the figure size

    for i, model_name in enumerate(model_list):
        model_name_clean = model_name.replace(".", "").split(":")[0]
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # Filter data for the current model
        model_data = aggregated_data[~aggregated_data['mean'].isna()]
        model_data = model_data[model_data['model'] == model_name_clean]
        model_data = model_data[['model', 'bin', "code system", 'mean', 'count', 'sem']]
        model_data['mean'] = model_data['mean'] 
        model_data = model_data[model_data['count'] >= 10]
        model_data = model_data.reset_index(drop=True)

        # Rename 'mean' column to fmatch for plotting
        fmatch = model_name_clean + "_Full_Code"
        model_data.rename(columns={'mean': fmatch}, inplace=True)
    
        # Create a bar plot
        barplot = sns.barplot(x='bin', y=fmatch, hue='code system', data=model_data, ax=ax, errorbar=None)
        
        for j in range(len(barplot.patches)):
            patch = barplot.patches[j]
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            if j < len(model_data):
                sem = model_data.iloc[j % len(model_data)]['sem']
                ax.errorbar(x, y, yerr=sem, fmt='none', color='black', capsize=5)
        
        for j, p in enumerate(barplot.patches):
            if p.get_width() != 0:
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                
                sem = model_data.iloc[j]['sem']
                ax.errorbar(x, y, yerr=sem, fmt='none', color='black', capsize=5)
                
                #count = model_data.iloc[j]['count']
                #ax.text(x,  + 0.01, count, ha='center', va='bottom', color='black', fontsize=10)
            
        ax.set_xlabel('Annual MSHS Code Frequency')
        ax.set_ylabel('Exact Code Match Rate')
        ax.set_title(f'{model_name_clean}')
        ax.legend(loc='upper left')
        ax.set_ylim(0, 1)

    # Adjust layout and save plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25) 
    plt.savefig(f"Output/combined_match_rate_bar_chart{suffix}.png", format='png', dpi=300)

    # Show the plot
    plt.show()
    
def manual_analysis(filepath1, filepath2):
    df_dict = {}
    results_dict = {}
    analysis_dict ={}
    
    with open(f"Output/Intermediate/df_analysis_man_automatedscores.pkl", "rb") as file:
        code_datasets = pickle.load(file)
        
    for codesystem in ["ICD9CM", "ICD10CM", "CPT"]:
        df_merge = pd.DataFrame()
        df_dict[codesystem] = {}
        df1 = pd.read_excel(filepath1,sheet_name=codesystem)
        df2 = pd.read_excel(filepath2,sheet_name=codesystem)
        
        df_merge = pd.merge(df1,df2, on=['codedesc','desc'], how='outer')
        df_merge = df_merge.rename(columns={"code": "Eyal", "Column1": "Ali"})
        df_merge['Eyal'] = df_merge['Eyal'].astype(int)
        df_merge['Ali'] = df_merge['Ali'].astype(int)
        df_merge['Avg_STS'] = (df_merge['Eyal'] + df_merge['Ali'] )/ 2
        
        for col in ['Eyal','Ali']:
            df_merge.loc[(df_merge[col] == 0) , f'{col}_simple'] = 0
            df_merge.loc[(df_merge[col] >= 1) & (df_merge[col] <= 3), f'{col}_simple'] = 1
            df_merge.loc[(df_merge[col] >= 4) , f'{col}_simple'] = 2
        
        df_merge['Avg_STS_simple'] = (df_merge['Eyal_simple'] + df_merge['Ali_simple'] )/ 2
        
        df_orig = pd.read_parquet(f"Output/Intermediate/{codesystem}_parsed_man.parquet")
        
        model_list = ["gpt-35-turbo-0301",
              "gpt-35-turbo-0613",
              "gpt-35-turbo-1106",
              "gpt-4-0314",
              "gpt-4-0613",
              "gpt-4-1106-preview",
              "gemini-pro",
              "meta/llama-2-70b-chat",
              ]
        
        df_results = code_datasets[codesystem].copy()
        df_results = df_results.loc[:,~df_results.columns.duplicated()]
        
        scores_list = []
        analysis_dict[codesystem] = {'Avg_STS': {}, 'Avg_STS_simple':{}, 'cui2vec':{},'cui2vec vectors':{}, 'icdcodex': {},'icdcodex vectors': {}, 'meteor15':{},'meteor15 vectors':{}, 'bertscore':{},'bertscore vectors':{}}
        
        # add manual scores to score datasets. 
        for model_name in model_list:

            manual_columns = [f'{codesystem}_codedesc',
                        f'{model_name}_desc',
                         f'{model_name}_Eyal', 
                         f'{model_name}_Eyal_simple', 
                         f'{model_name}_Ali', 
                         f'{model_name}_Ali_simple', 
                         f'{model_name}_Avg_STS',
                         f'{model_name}_Avg_STS_simple',
                         f'{model_name}_cui2vec', 
                         f'{model_name}_icdcodex', 
                         f'{model_name}_meteor15', 
                         f'{model_name}_bertscore']
            
            results_columns = [f'{codesystem}_codedesc',
                            f'{model_name}_desc' ,
                            f'{model_name}_cui2vec', 
                            f'{model_name}_icdcodex', 
                            f'{model_name}_meteor15', 
                            f'{model_name}_bertscore']
        
            orig_columns = [f'{codesystem}_code', 
                        f'{codesystem}_codedesc',
                        f'{codesystem}_count',
                        'probability',
                        f'{model_name}',
                        f'{model_name}_desc']
                
            df_merge_model = df_merge.rename(columns={'codedesc':f'{codesystem}_codedesc',
                                                    'desc':f'{model_name}_desc',
                                                    "Eyal": f"{model_name}_Eyal", 
                                                    "Eyal_simple": f"{model_name}_Eyal_simple", 
                                                    "Ali":f"{model_name}_Ali", 
                                                    "Ali_simple":f"{model_name}_Ali_simple", 
                                                    "Avg_STS":f'{model_name}_Avg_STS',
                                                    "Avg_STS_simple":f'{model_name}_Avg_STS_simple'})
        
            df_merge_model = pd.merge(df_merge_model, df_results[results_columns], on=[f'{codesystem}_codedesc',f'{model_name}_desc'], how="inner")
            
            scores_list.extend(zip(*[df_merge_model[col] for col in manual_columns]))
            
            manual_col_simple = [col.split("_", 1)[1] for col in manual_columns]

            df_final = pd.merge(df_orig[orig_columns], df_merge_model[manual_columns], on=[f'{codesystem}_codedesc',f'{model_name}_desc'], how="left")
        
            metric_dict = {"Avg_STS":f'{model_name}_Avg_STS',
                           "Avg_STS_simple":f'{model_name}_Avg_STS_simple',
                           "cui2vec":f'{model_name}_cui2vec',
                           "icdcodex":f'{model_name}_icdcodex',
                           "meteor15":f'{model_name}_meteor15',
                           "bertscore":f'{model_name}_bertscore'}
        
            df_final.loc[(df_final[f'{model_name}_desc'] == df_final[f'{codesystem}_codedesc']) , f'{model_name}_Eyal'] = 5
            df_final.loc[(df_final[f'{model_name}_desc'] == df_final[f'{codesystem}_codedesc']) , f'{model_name}_Ali'] = 5
            df_final.loc[(df_final[f'{model_name}_desc'] == df_final[f'{codesystem}_codedesc']) , f'{model_name}_Avg_STS'] = 5
            df_final.loc[(df_final[f'{model_name}_desc'] == df_final[f'{codesystem}_codedesc']) , f'{model_name}_Eyal_simple'] = 3
            df_final.loc[(df_final[f'{model_name}_desc'] == df_final[f'{codesystem}_codedesc']) , f'{model_name}_Ali_simple'] = 3
            df_final.loc[(df_final[f'{model_name}_desc'] == df_final[f'{codesystem}_codedesc']) , f'{model_name}_Avg_STS_simple'] = 3
        
            for metric, metric_col in metric_dict.items():
                if "STS" not in metric:
                    df_final.loc[(df_final[f'{model_name}_desc'] == df_final[f'{codesystem}_codedesc']) , metric_col] = 1
                    
                analysis_dict[codesystem][metric][model_name] = str(round(df_final[metric_col].mean(),3))
                if "STS" not in metric:
                    analysis_dict[codesystem][f'{metric} vectors'][model_name] = str(df_final[metric_col].count())
            
            df_dict[codesystem][model_name] = df_final
        
        ## CORRELATION CALCULATION ##
        # create a scores dataframe with unique description pairs
        df_results_all = pd.DataFrame()
        df_results_all[manual_col_simple] = pd.DataFrame(scores_list, columns=manual_col_simple)
        df_results_all.drop_duplicates(inplace=True)
        df_results_all['codedesc'].dropna(inplace=True)
        df_results_all = df_results_all[df_results_all['codedesc'] != df_results_all['desc']]
        
        kappa = cohen_kappa_score(df_merge['Eyal'], df_merge['Ali'])
        kappa_simple = cohen_kappa_score(df_merge['Eyal_simple'], df_merge['Ali_simple'])
      
        # Calculate Pearson correlation between manual and automated scores
        correlation_dict ={}
        for var in ['Avg_STS', "Avg_STS_simple", "Ali_simple", "Eyal_simple"]:
            if "ICD" in codesystem:
                correlation_matrix = df_results_all[[var, 'cui2vec', 'icdcodex', 'meteor15', 'bertscore']].corr(method='pearson')
                correlation = correlation_matrix.loc[var, ['cui2vec', 'icdcodex', 'meteor15', 'bertscore']]
                            
            else:
                correlation_matrix = df_results_all[['Avg_STS', 'meteor15', 'bertscore']].corr(method='pearson')
                correlation = correlation_matrix.loc[f'Avg_STS', ['meteor15', 'bertscore']]
            
            correlation_dict[var] = correlation
        
        results_dict[codesystem] = {"Kappa":round(kappa,3), "Correlation":correlation_dict['Avg_STS'], 
                                    "Kappa, simple":round(kappa_simple,3), "Correlation, simple":correlation_dict['Avg_STS_simple'],
                                    "Correlation, Ali":correlation_dict['Ali_simple'], "Correlation, Eyal":correlation_dict['Eyal_simple']} 
        
        with open(f"Output/Intermediate/df_analysis_manauto.pkl", "wb") as file:
            pickle.dump(df_dict, file)
        
        with open(f"Output/Intermediate/analysis_manauto.json", "w") as file:
            json.dump(analysis_dict, file)
        
    return results_dict, df_dict

def create_metric_table_old(suffix):
    ##load data
    # simple metrics
    with open(f"Output/Intermediate/df_analysis{suffix}.pkl", "rb") as file:
        df_results1 = pickle.load(file)
        
    with open(f"Output/Intermediate/analysis{suffix}.json", "r") as file:
        result_dict1 = json.load(file)

    # automated metrics
    if suffix == "_man":
        with open(f"Output/Intermediate/df_analysis_manauto.pkl", "rb") as file:
            df_results2 = pickle.load(file)
        
        with open(f"Output/Intermediate/analysis_manauto.json", "r") as file:
            result_dict2 = json.load(file)
        
    if suffix == "":
        with open(f"Output/Intermediate/df_analysis_automatedscores.pkl", "rb") as file:
            df_results2 = pickle.load(file)
            
        with open(f"Output/Intermediate/analysis_automatedscores.json", "r") as file:
            result_dict2 = json.load(file)

    model_list = ["gpt-35-turbo-0301",
                "gpt-35-turbo-0613",
                "gpt-35-turbo-1106",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-1106-preview",
                "gemini-pro",
                "meta/llama-2-70b-chat",
                ]
    
    # merge data
    df_merge = {}
    for codesystem in ["ICD9CM", "ICD10CM", "CPT"]:
        df_merge[codesystem] = {}

        for model_name in model_list:
            if suffix == "_man":
                col_list2 = [f'{codesystem}_code',
                            f'{codesystem}_codedesc',
                            f'{model_name}',
                            f'{model_name}_desc',
                            f'{model_name}_Avg_STS',
                            f'{model_name}_cui2vec', 
                            f'{model_name}_icdcodex', 
                            f'{model_name}_meteor15', 
                            f'{model_name}_bertscore']
                
            if suffix == "":
                col_list2 = [f'{codesystem}_code',
                            f'{codesystem}_codedesc',
                            f'{model_name}',
                            f'{model_name}_desc',
                            f'{model_name}_cui2vec', 
                            f'{model_name}_icdcodex', 
                            f'{model_name}_meteor15', 
                            f'{model_name}_bertscore']

            if codesystem != "CPT":
                col_list1 = [f'{codesystem}_code',
                            f'{model_name}',
                            f'{model_name}_ExactMatch',
                            f'{model_name}_BillableCode',
                            f'{model_name}_LengthMatch', 
                            f'{model_name}_DigitMatch_Ovr', 
                            f'{model_name}_DigitMatch_Ovr_list']
            else:
                col_list1 = [f'{codesystem}_code',
                            f'{model_name}',
                            f'{model_name}_ExactMatch',
                            f'{model_name}_LengthMatch', 
                            f'{model_name}_DigitMatch_Ovr', 
                            f'{model_name}_DigitMatch_Ovr_list']

            if suffix == "_man":
                merged_df = pd.merge(df_results1[codesystem][col_list1], 
                                    df_results2[codesystem][model_name][col_list2], 
                                    on=[f'{codesystem}_code', model_name], 
                                    how="inner")
            if suffix == "":
                merged_df = pd.merge(df_results1[codesystem][col_list1], 
                                    df_results2[codesystem][col_list2], 
                                    on=[f'{codesystem}_code', model_name], 
                                    how="inner")
        
            df_merge[codesystem][model_name] = merged_df

    for codesystem in result_dict2:
        for metric in result_dict2[codesystem]:
            result_dict1[codesystem][metric] = result_dict2[codesystem][metric]
    
    metrics = {}
    models = set()
    for codesystem, metrics in result_dict1.items():
        for metric, values in metrics.items():
            if not metric.endswith("vectors"):  # Exclude vector count entries for column headers
                models.update(values.keys())

    # Convert the set of models to a list to maintain order
    models_list = sorted(list(models))

    # Initialize an empty DataFrame
    df = pd.DataFrame(index=pd.MultiIndex.from_product([result_dict1.keys(), metrics.keys()], names=['Code System', 'Metric']), columns=models_list)

    # Populate Table
    for codesystem, metrics in result_dict1.items():
        for metric, values in metrics.items():
            if metric.endswith("vectors") or metric.endswith("simple"):
                continue
            vector_metric = metric + " vectors"  # Assuming vector counts follow this naming convention
            for model, score in values.items():
                # Retrieve corresponding vector count
                vector_count = result_dict1[codesystem].get(vector_metric, {}).get(model, None)
                total_count = df_merge[codesystem][model][f'{model}_desc'].count()
                
                # Format cell as "score (vector count / valid codes)"
                cell_value = f"{score} ({round(int(vector_count)/total_count*100,1)}%)" if vector_count is not None else str(score)
                df.at[(codesystem, metric), model] = cell_value
            
    df = df[~df.map(lambda x: 'nan' in str(x)).any(axis=1)]

    # Optional: formatting for publication (adjust as needed)
    styled_df = df.style.set_table_styles([{
        'selector': 'th',
        'props': [('font-size', '10pt'), ('text-align', 'center')]
        }])
    styled_df= styled_df.set_properties(**{
                    'text-align': 'center',
                    'font-size': '9pt'
                        })
    
    styled_df.to_excel(f"Output/metrics_tables{suffix}.xlsx")
    
    return df

def create_metric_table(suffix, error_analysis=False):
    ##load data
    # simple metrics
    with open(f"Output/Intermediate/df_analysis{suffix}.pkl", "rb") as file:
        df_results1 = pickle.load(file)
        
    # automated metrics
    if suffix == "_man":
        with open(f"Output/Intermediate/df_analysis_manauto.pkl", "rb") as file:
            df_results2 = pickle.load(file)
        
    if suffix == "":
        with open(f"Output/Intermediate/df_analysis_automatedscores.pkl", "rb") as file:
            df_results2 = pickle.load(file)

    # merge data
    df_merge = {}
    result_dict1 = {}
    model_list = ["gpt-35-turbo-0301",
                "gpt-35-turbo-0613",
                "gpt-35-turbo-1106",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-1106-preview",
                "gemini-pro",
                "meta/llama-2-70b-chat",
                ]
    
    for codesystem in ["ICD9CM", "ICD10CM", "CPT"]:
        df_merge[codesystem] = {}
        result_dict1[codesystem] = {'ExactMatch':{},
                                'BillableCode':{},
                                'LengthMatch':{}, 
                                'DigitMatch_Ovr':{}, 
                                'DigitMatch_Ind':{},
                                'Avg_STS': {}, 
                                'Avg_STS_simple':{}, 
                                'cui2vec':{},
                                'cui2vec vectors':{}, 
                                'icdcodex': {},
                                'icdcodex vectors': {}, 
                                'meteor15':{},
                                'meteor15 vectors':{}, 
                                'bertscore':{},
                                'bertscore vectors':{}}

        for model_name in model_list:
            #Column List 1
            col_list1 = [f'{codesystem}_code',
                            f'{model_name}',
                            f'{model_name}_ExactMatch']

            if codesystem != "CPT":
                col_list1.append(f'{model_name}_BillableCode')
                
            col_list1.extend([
                            f'{model_name}_LengthMatch', 
                            f'{model_name}_DigitMatch_Ovr', 
                            f'{model_name}_DigitMatch_Ovr_list',
                            'code_length'])
            
            #Column List 1
            col_list2 = [f'{codesystem}_code',
                            f'{codesystem}_codedesc',
                            f'{model_name}',
                            f'{model_name}_desc']
            
            if suffix == "_man":
                col_list2.append(f'{model_name}_Avg_STS')
            
            col_list2.extend([f'{model_name}_cui2vec', 
                            f'{model_name}_icdcodex', 
                            f'{model_name}_meteor15', 
                            f'{model_name}_bertscore'])
            
            if suffix == "_man":
                merged_df = pd.merge(df_results1[codesystem][col_list1], 
                                    df_results2[codesystem][model_name][col_list2], 
                                    on=[f'{codesystem}_code', model_name], 
                                    how="inner")
            if suffix == "":
                merged_df = pd.merge(df_results1[codesystem][col_list1], 
                                    df_results2[codesystem][col_list2], 
                                    on=[f'{codesystem}_code', model_name], 
                                    how="inner")
            
            if error_analysis == True:            
                merged_df = merged_df[merged_df[f'{codesystem}_codedesc'] != merged_df[f'{model_name}_desc']]
            
            df_merge[codesystem][model_name] = merged_df
            
            # Metric List
            metric_dict ={'ExactMatch':f'{model_name}_ExactMatch'}
            
            if codesystem!= "CPT":
                metric_dict.update({'BillableCode':f'{model_name}_BillableCode'})
            
            metric_dict.update({'LengthMatch':f'{model_name}_LengthMatch', 
                            'DigitMatch_Ovr':f'{model_name}_DigitMatch_Ovr', 
                            'DigitMatch_Ovr_list':f'{model_name}_DigitMatch_Ovr_list'})
            
            if suffix=="_man":
                metric_dict.update({'Avg_STS':f'{model_name}_Avg_STS'})
            
            Automated_metrics = {'cui2vec':f'{model_name}_cui2vec', 
                            'icdcodex':f'{model_name}_icdcodex', 
                            'meteor15':f'{model_name}_meteor15',
                            'bertscore':f'{model_name}_bertscore'}
            
            metric_dict.update(Automated_metrics)
            
            for metric, metric_col in metric_dict.items():
                if metric in ['ExactMatch','BillableCode','LengthMatch','DigitMatch_Ovr']:
                    value = str(round(merged_df[metric_col].mean()*100,1)) + "%"
                    if error_analysis==True and metric =='ExactMatch':
                        continue
                    result_dict1[codesystem][metric][model_name] = value
                if metric == "DigitMatch_Ovr_list":
                    digit_match = merged_df[metric_col].explode()
                    result_dict1[codesystem]['DigitMatch_Ind'][model_name] = {
                        key: (str(round(value / (merged_df['code_length'] >= key).sum() * 100, 1)) + "%")
                        if (merged_df['code_length'] >= key).sum() != 0 
                        else '0.0%' 
                        for key, value in digit_match.value_counts().items()
                        }
                elif metric == "Avg_STS":
                    value = round(merged_df[metric_col].mean(),1)
                    result_dict1[codesystem][metric][model_name] = value
                elif metric in Automated_metrics.keys():
                    vector_count = merged_df[metric_col].count()
                    value = round(merged_df[metric_col].mean(),3)
                    result_dict1[codesystem][metric][model_name] = value
                    result_dict1[codesystem][f'{metric} vectors'][model_name] = vector_count

    metrics = {}
    models = set()
    for codesystem, metrics in result_dict1.items():
        for metric, values in metrics.items():
            if not metric.endswith("vectors"):  # Exclude vector count entries for column headers
                models.update(values.keys())

    # Convert the set of models to a list to maintain order
    models_list = sorted(list(models))

    # Initialize an empty DataFrame
    df = pd.DataFrame(index=pd.MultiIndex.from_product([result_dict1.keys(), metrics.keys()], names=['Code System', 'Metric']), columns=models_list)

    # Populate Table
    for codesystem, metrics in result_dict1.items():
        for metric, values in metrics.items():
            if metric.endswith("vectors") or metric.endswith("simple"):
                continue
            vector_metric = metric + " vectors"  # Assuming vector counts follow this naming convention                
            for model, score in values.items():
                if metric.endswith("Ind"):
                    sorted_digit_match = sorted(score.items(), key=lambda x: x[0])
                    score = "\n".join([f"{key}: {value}" for key, value in sorted_digit_match])
                # Retrieve corresponding vector count
                vector_count = None
                #vector_count = result_dict1[codesystem].get(vector_metric, {}).get(model, None)
                total_count = df_merge[codesystem][model][f'{model}_desc'].count()
                
                # Format cell as "score (vector count / valid codes)"
                cell_value = f"{score} ({round(int(vector_count)/total_count*100,1)}%)" if vector_count is not None else str(score)
                if (metric.endswith("cui2vec") or metric.endswith("icdcodex")) and codesystem == "CPT":
                    continue
                df.at[(codesystem, metric), model] = cell_value
            
    df = df[~df.map(lambda x: 'nan' in str(x)).any(axis=1)]

    # Optional: formatting for publication (adjust as needed)
    styled_df = df.style.set_table_styles([{
        'selector': 'th',
        'props': [('font-size', '10pt'), ('text-align', 'center')]
        }])
    styled_df= styled_df.set_properties(**{
                    'text-align': 'center',
                    'font-size': '9pt'
                        })
    if error_analysis == True:
        styled_df.to_excel(f"Output/metrics_tables{suffix}_nomatch.xlsx")   
    else:   
        styled_df.to_excel(f"Output/metrics_tables{suffix}.xlsx")
    
    return df

def sts_score_dist_fig(error_analysis=False):
    # Load your DataFrame
    with open(f"Output/Intermediate/df_analysis_manauto.pkl", "rb") as file:
        df_dict = pickle.load(file)

    # Define the models and code systems
    models = [
        "gpt-35-turbo-0301",
        "gpt-35-turbo-0613",
        "gpt-35-turbo-1106",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-1106-preview",
        "gemini-pro",
        "meta/llama-2-70b-chat",
    ]
    code_systems = ['ICD9CM', 'ICD10CM', 'CPT']

    model_dict = {"gpt-35-turbo-0301":"gpt-3.5-turbo-0301",
        "gpt-35-turbo-0613":"gpt-3.5-turbo-0613",
        "gpt-35-turbo-1106":"gpt-3.5-turbo-1106",
        "meta/llama-2-70b-chat":"llama-2-70b-chat"}

    # Plot settings
    num_models = len(models)
    cols = 4  # Adjust as needed
    rows_per_system = num_models // cols + (num_models % cols > 0)

    # Determine common axis limits (optional, adjust as needed)
    x_min, x_max = 0, 5  

    # Iterate over each code system and model
    for codesystem in code_systems:
        plt.figure(figsize=(12, 2 * rows_per_system))  # Adjust figure size as needed

        for i, model in enumerate(models):
            if model in df_dict[codesystem]:
                df = df_dict[codesystem][model]
                if error_analysis == True:            
                    df = df[df[f'{codesystem}_codedesc'] != df[f'{model}_desc']]
                columns_with_avg_STS = [col for col in df.columns if 'Avg_STS' in col and "simple" not in col]

                # Aggregate Avg_STS scores from all columns for this model
                all_avg_STS_scores = df[columns_with_avg_STS].values.flatten()
                all_avg_STS_scores = all_avg_STS_scores[~pd.isna(all_avg_STS_scores)]  # Remove NaN values
            
                model = model_dict.get(model, model)

                # Plotting the histogram in a subplot
                plt.subplot(rows_per_system, cols, i + 1)
                plt.hist(all_avg_STS_scores, bins=20, edgecolor='black')  # Adjust bins as needed
                plt.title(model)
                plt.xlabel('Average STS Score')
                plt.ylabel('Frequency')
                plt.xlim(x_min, x_max)
        
        if error_analysis == True:
            suffix = "_error"
        else:
            suffix = ""
        
        plt.suptitle(f'Distribution of {codesystem} STS Scores for Each Model')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout with room for subtitle
        plt.savefig(f"Output/{codesystem}_STS_Score_distributions{suffix}.png", format='png', dpi=300)
        plt.show()

def code_features_dist(suffix=""):
    with open(f"Output/Intermediate/df_analysis{suffix}.pkl", "rb") as file:
        df_results = pickle.load(file)

    model_list = ["gpt-35-turbo-0301",
                    "gpt-35-turbo-0613",
                    "gpt-35-turbo-1106",
                    "gpt-4-0314",
                    "gpt-4-0613",
                    "gpt-4-1106-preview",
                    "gemini-pro",
                    "meta/llama-2-70b-chat",
                    ]
    codesystems = ['ICD9CM', 'ICD10CM', 'CPT']

    # Create the figure with subplots    
    n_models = len(model_list)  

    for codesystem in codesystems:
        if codesystem == "ICD9CM":
            height_ratios = [1.0, 2.0, 4.0]  # Height ratios for 3 rows
        elif codesystem == "ICD10CM":
            height_ratios = [1.4, 2.0, 4.0]  # Height ratios for 3 rows
        if codesystem == "CPT":
            height_ratios = [0.5, 2.2, 5.0]  # Height ratios for 3 rows
            
        gs = gridspec.GridSpec(nrows=3, ncols=n_models, height_ratios=height_ratios)

        fig = plt.figure(figsize=(n_models * 3, sum(height_ratios) * 1))  # Adjust the figsize
        
        df_results[codesystem]['description_length'] = df_results[codesystem][f'{codesystem}_codedesc'].str.len()
        if codesystem=="CPT":
            df_results[codesystem]['description_length_25'] = df_results[codesystem]['description_length']/20  
            divide_25 = [(i, i + 20) for i in range(0, 2001, 20)]
            df_results[codesystem]['description_length_bin'] = [val for value in df_results[codesystem]['description_length'] for (lower_bound, upper_bound), val in zip(divide_25, range(0, 2001, 20)) if lower_bound <= value < upper_bound]
            df_results[codesystem]['description_length_bin'] = df_results[codesystem]['description_length_bin'].apply(lambda x: x if x <= 400 else 400)
        else:
            df_results[codesystem]['description_length_10'] = df_results[codesystem]['description_length']/10    
            divide_10 = [(i, i + 10) for i in range(0, 401, 10)]
            df_results[codesystem]['description_length_bin'] = [val for value in df_results[codesystem]['description_length'] for (lower_bound, upper_bound), val in zip(divide_10, range(0, 401, 10)) if lower_bound <= value < upper_bound]
            df_results[codesystem]['description_length_bin'] = df_results[codesystem]['description_length_bin'].apply(lambda x: x if x <= 150 else 150)
            
        df_results[codesystem][f'{codesystem}_count_log'] = np.log10(df_results[codesystem][f'{codesystem}_count'])
        df_results[codesystem][f'{codesystem}_count_log_bin'] = np.searchsorted([10**i for i in range(9)], df_results[codesystem][f'{codesystem}_count'])
        
        for i, feature in enumerate(['code_length', f'{codesystem}_count_log_bin', 'description_length_bin']):
            for j, model_name in enumerate(model_list):
                ax = plt.subplot(gs[i, j])  
                df_regress = df_results[codesystem].copy()
                df_regress['ExactMatch'] = df_regress[f'{model_name}_ExactMatch']
                
                df_match_N = df_regress[df_regress['ExactMatch'] == 0]
                df_match_Y = df_regress[df_regress['ExactMatch'] == 1]
                
                dim_dict = {"":{("ICD9CM","code_length"):(6000,4000,2000),
                                ("ICD9CM",'ICD9CM_count_log_bin'):(3000,3000,1500),
                                ("ICD9CM",'description_length_bin'):(2000,2000,1000),
                                ("ICD10CM","code_length"):(5000,5000,2500),
                                ("ICD10CM",'ICD10CM_count_log_bin'):(9000,6000,3000),
                                ("ICD10CM",'description_length_bin'):(3000,3000,1500),
                                ("CPT","code_length"):(4500,4500,1500),
                                ("CPT",'CPT_count_log_bin'):(1500,1000,500),
                                ("CPT",'description_length_bin'):(750,500,250)
                                },
                            "_man":{('ICD9CM', 'code_length'): (100, 150, 50),
                                    ('ICD9CM', 'ICD9CM_count_log_bin'): (100, 150, 50),
                                    ('ICD9CM', 'description_length_bin'): (75, 75, 25),
                                    ('ICD10CM', 'code_length'): (100, 150, 50),
                                    ('ICD10CM', 'ICD10CM_count_log_bin'): (100, 150, 50),
                                    ('ICD10CM', 'description_length_bin'): (50, 75, 25),
                                    ('CPT', 'code_length'): (250, 250, 125),
                                    ('CPT', 'CPT_count_log_bin'): (100, 150, 50),
                                    ('CPT', 'description_length_bin'): (60, 60, 30)}}
                
                
                max_count_N, max_count_Y, tick_interval = dim_dict[suffix][(codesystem,feature)]
                
                sorted_categories = sorted(df_regress[feature].unique(), reverse=True)
                if codesystem == "CPT" and feature == "code_length":
                    bar_width = 0.5
                else:
                    bar_width = 0.9
                
                # Count the occurrences for each category
                count_0 = df_match_N[feature].value_counts().reindex(sorted_categories, fill_value=0)
                count_1 = df_match_Y[feature].value_counts().reindex(sorted_categories, fill_value=0)
                
                ratio={}
                for category in sorted_categories:
                    if count_0[category] == 0 and count_1[category] > 0:
                        ratio[str(category)] = 1.0
                    elif count_1[category] == 0 and count_0[category] > 0:
                        ratio[str(category)] = 0.0
                    else:
                        total_count = count_0[category] + count_1[category]
                        ratio[str(category)] = count_1[category] / total_count if total_count > 0 else 0.0
                
                # Plotting for ExactMatch == 1
                sns.countplot(y=feature, data=df_match_Y, ax=ax, color='green', order=sorted_categories, width=bar_width) 
                ax.set_xlim(-max_count_N, max_count_Y)  
                # Set x-ticks after defining the limits
                ticks_Y = np.arange(0, max_count_Y + tick_interval, tick_interval)
                negative_ticks_Y = -np.arange(0, max_count_N + tick_interval, tick_interval)[1:]
                combined_ticks_Y = np.concatenate([negative_ticks_Y, ticks_Y])

                ax.set_xlim(-max_count_N, max_count_Y)
                ax.set_xticks(combined_ticks_Y)
                ax.set_xticklabels([f"{int(abs(tick))}" for tick in combined_ticks_Y])

                ax.set_xlabel('Code Count')
                ax.set_ylabel('')
                
                # Creating an inverted plot for ExactMatch == 0
                ax1 = ax.twiny()  
                sns.countplot(y=feature, data=df_match_N, ax=ax1, color='red', order=sorted_categories, width=bar_width)
                
                ticks_N = np.arange(0, max_count_N + tick_interval, tick_interval)
                negative_ticks_N = -np.arange(0, max_count_Y + tick_interval, tick_interval)[1:]
                combined_ticks_N = np.concatenate([negative_ticks_N, ticks_N])

                ax1.set_xlim(-max_count_Y, max_count_N)
                ax1.invert_xaxis()
                ax1.set_xticks(combined_ticks_N)
                ax1.set_xticklabels([f"{int(abs(tick))}" for tick in combined_ticks_N])
                ax1.set_xlabel('')
                
                sorted_categories = sorted(df_regress[feature].unique(), reverse=True)
                
                for a,category in enumerate(sorted_categories):
                    # Find patches in both ax and ax1 that correspond to the current category
                    patches_ax = [p for p in ax.patches if ax.get_yticklabels()[int(p.get_y() + p.get_height() / 2)].get_text() == str(category)]
                    patches_ax1 = [p for p in ax1.patches if ax1.get_yticklabels()[int(p.get_y() + p.get_height() / 2)].get_text() == str(category)]

                    # Determine which patch to use
                    if patches_ax:
                        patch = patches_ax[0]
                        width = patch.get_width()
                    elif patches_ax1:
                        patch = patches_ax1[0]
                        if count_1[category] > 0:
                            pass
                            last_category = sorted_categories[a-1]
                            last_patches_ax = [p for p in ax.patches if ax.get_yticklabels()[int(p.get_y() + p.get_height() / 2)].get_text() == str(last_category)]
                            last_patch = last_patches_ax[0]
                            width = count_1[category] / count_1[last_category] * last_patch.get_width()
                        else:
                            width = 0
                    else:
                        # Skip this category if there are no patches in either axis
                        continue
                    
                    
                    
                    height = patch.get_height()
                    y = patch.get_y()

                    # Use the ratio dictionary to get the value
                    value = ratio.get(str(category), 0.0)  # Default to 0 if the category is not in the ratio dict

                    ax.annotate(f'{float(value):.2f}',  # Format to two decimal places
                        (patch.get_x() + width, y + height / 2), 
                        xytext=(5, 0),  # 5 points offset
                        textcoords='offset points', 
                        ha='left', 
                        va='center')

                # Set titles, labels, etc.
                if i == 0:
                    ax.set_title(model_name)    
                if j == 0:
                    if feature == "code_length":
                        ax.set_ylabel("Code Length\n(digits)")
                    elif feature == f'{codesystem}_count_log_bin':
                        ax.set_ylabel("Code Frequency\n(log)")
                    elif feature == 'description_length_bin':
                        ax.set_ylabel("Description Length\n(characters)")

            # # Logistic Regression Analysis
            # if codesystem=="CPT":
            #     features = [f'{codesystem}_count_log', 'code_length', 'description_length_25']
            # else:
            #     features = [f'{codesystem}_count_log', 'code_length', 'description_length_10']
            # X = df_regress[features]
            # y = df_regress['ExactMatch']
            # model = LogisticRegression()
            # model.fit(X, y)
            
            # # Getting the estimated coefficients
            # coefficients = model.coef_[0]

            # # Calculating odds ratios
            # odds_ratios = np.exp(coefficients)

            # # Calculating standard errors and other statistics
            # standard_errors = np.sqrt(np.diag(np.linalg.inv(np.dot(X.T, X))))
            # wald_stats = coefficients / standard_errors
            # p_values = chi2.sf(wald_stats**2, 1)
            # conf_intervals = pd.DataFrame(index=features, columns=["lower", "upper"])
            # for i in range(X.shape[1]):
            #     conf_intervals.iloc[i] = np.exp(coefficients[i] - 1.96 * standard_errors[i]), np.exp(coefficients[i] + 1.96 * standard_errors[i])

            # odds_ratios_with_confidence_intervals = pd.DataFrame({
            #     "Odds Ratio": odds_ratios,
            #     "Lower CI (95%)": conf_intervals["lower"],
            #     "Upper CI (95%)": conf_intervals["upper"],
            #     "p-value": p_values
            # }, index=features)

            # print(f"Code System: {codesystem}, Model: {model_name}")
            # print(odds_ratios_with_confidence_intervals)
            # print()
            
        print(codesystem)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect if the title is overlapping
        plt.savefig(f"Output/{codesystem}_MetricDistribution_{suffix}.png", format='png', dpi=300)
        plt.show()