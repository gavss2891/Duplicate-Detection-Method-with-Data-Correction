import json
import os
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from sympy import divisors
from multiprocessing import Pool, cpu_count
import data_cleaning
import lsh
import MSM as msm

def find_predictions(clusters):
    predictions = set()
    labels = clusters.labels_
    for i in range(clusters.n_clusters_):
        contained_in_i = np.where(labels == i)[0]
        if len(contained_in_i) > 1:
            predictions.update(combinations(contained_in_i, 2))
    return predictions

def check_duplicates(data, dissimilarity, candidates, clusters):
    duplicates = set()
    predictions_lsh = set()
    duplicates_initial = defaultdict(list)
    
    product_names = dissimilarity.columns
    predictions = find_predictions(clusters)
    
    for index in range(len(product_names)):
        model_id = data.get(product_names[index]).get("modelID")
        duplicates_initial[model_id].append(index)
        for index_2 in range(index + 1, len(product_names)):
            if candidates.iloc[index, index_2] == 0:
                predictions_lsh.add((index, index_2))

    for value in duplicates_initial.values():
        if len(value) >= 2:
            duplicates.update(combinations(value, 2))
    
    TP_set = predictions.intersection(duplicates)
    FP_set = predictions.difference(duplicates)
    FN_set = duplicates.difference(predictions)
    DF_set = predictions_lsh.intersection(duplicates)

    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)
    DF = len(DF_set)

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    np.fill_diagonal(candidates.values, 1)
    n_comp = ((candidates == 0).sum().sum()) / 2
    n_possible_comp = candidates.shape[0] * (candidates.shape[0] - 1) / 2
    frac_comp = n_comp / n_possible_comp
   
    PQ = DF / n_comp if n_comp != 0 else 0
    PC = DF / len(duplicates) if len(duplicates) > 0 else 0

    F_1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    F_1_star = (2 * PQ * PC) / (PQ + PC) if PQ + PC != 0 else 0

    return (F_1, F_1_star, PC, PQ, n_comp, frac_comp, precision, recall)

def plot_metrics(dataframe_corrected, dataframe_no_correction, metric1, metric2, path):
    os.makedirs(path, exist_ok=True)
    grouped_corrected = dataframe_corrected.groupby(metric1, as_index=False).agg({metric2: 'max'})
    grouped_no_correction = dataframe_no_correction.groupby(metric1, as_index=False).agg({metric2: 'max'})
    title = f"{metric1} vs. {metric2}".replace("*", "star")
    filename = f"{title}.png"
    
    plt.figure(figsize=(8, 6))
    plt.plot(grouped_corrected[metric1], grouped_corrected[metric2], color='black', linestyle='-', label='CleanMSMP+')
    plt.plot(grouped_no_correction[metric1], grouped_no_correction[metric2], color='0.3', linestyle='--', label='MSMP+')
    
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    plt.legend()
    plt.savefig(os.path.join(path, filename))
    plt.close()

def save_checkpoint(checkpoint_dir, bootstrap_id, store_df, metrics_summary):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"bootstrap_{bootstrap_id}.pkl")
    checkpoint_data = {
        'results': store_df,
        'metrics_summary': metrics_summary,
        'bootstrap_id': bootstrap_id
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Saved checkpoint for bootstrap {bootstrap_id + 1}")

def load_checkpoint(checkpoint_dir, bootstrap_id):
    checkpoint_file = os.path.join(checkpoint_dir, f"bootstrap_{bootstrap_id}.pkl")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def get_completed_bootstraps(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return []
    return sorted([int(f.split("_")[1].split(".")[0]) 
                   for f in os.listdir(checkpoint_dir) 
                   if f.startswith("bootstrap_") and f.endswith(".pkl")])

def load_tv_brands():
    csv_path = os.path.join(os.path.dirname(__file__), 'television_brands.csv')
    with open(csv_path, newline='', encoding='utf-8') as f:
        return sorted([r['Brand'].strip().lower() for r in csv.DictReader(f) if r.get('Brand', '').strip()], 
                     key=len, reverse=True)

def get_factors(n):
    return sorted(divisors(n))

def calculate_metrics_summary(store_df, best_params=None):
    best_f1_idx = store_df['F1'].idxmax()
    summary = {
        'best_factor': best_f1_idx,
        'best_f1': store_df.loc[best_f1_idx, 'F1'],
        'best_f1_star': store_df.loc[best_f1_idx, 'F1*'],
        'best_pc': store_df.loc[best_f1_idx, 'PC'],
        'best_pq': store_df.loc[best_f1_idx, 'PQ'],
        'best_frac_comp': store_df.loc[best_f1_idx, 'Fraction Comparisons'],
        'best_threshold': store_df.loc[best_f1_idx, 'Threshold'],
        'best_n_comp': store_df.loc[best_f1_idx, 'Number Comparisons'],
        'avg_f1': store_df['F1'].mean(),
        'avg_f1_star': store_df['F1*'].mean(),
        'avg_pc': store_df['PC'].mean(),
        'avg_pq': store_df['PQ'].mean(),
        'avg_frac_comp': store_df['Fraction Comparisons'].mean(),
    }
    if best_params:
        summary['best_gamma'] = best_params['gamma']
        summary['best_epsilon'] = best_params['epsilon']
        summary['best_mu'] = best_params['mu']
    return summary

def print_metrics_summary(metrics_summary, bootstrap_id=None):
    title = "METRICS SUMMARY"
    if bootstrap_id is not None:
        title = f"BOOTSTRAP {bootstrap_id + 1} - {title}"
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"Best Factor (b): {metrics_summary['best_factor']}")
    print(f"Best Threshold: {metrics_summary['best_threshold']:.6f}")
    if 'best_gamma' in metrics_summary:
        print(f"Best Parameters: gamma={metrics_summary['best_gamma']:.2f}, epsilon={metrics_summary['best_epsilon']:.2f}, mu={metrics_summary['best_mu']:.2f}")
    print(f"\nBest Metrics:")
    print(f"  F1:            {metrics_summary['best_f1']:.4f}")
    print(f"  F1*:           {metrics_summary['best_f1_star']:.4f}")
    print(f"  PC:            {metrics_summary['best_pc']:.4f}")
    print(f"  PQ:            {metrics_summary['best_pq']:.4f}")
    print(f"  Fraction Comp: {metrics_summary['best_frac_comp']:.4f}")
    print(f"  N Comparisons: {metrics_summary['best_n_comp']:.0f}")
    print(f"{'='*60}\n")

def run_bootstrap(args):
    data, bootstrap_id, checkpoint_dir, params = args
    print(f"Running bootstrap {bootstrap_id + 1}\n")
    
    cleaned_data = data_cleaning.clean_data(data.copy(), data_correction=params['data_correction'])
    dup, non_dup = data_cleaning.separate_duplicates(cleaned_data)
    train, test = data_cleaning.test_train_split(dup, non_dup)
    
    tv_brands = load_tv_brands()
    msm.set_tv_brands(tv_brands)
    
    mw_train = lsh.get_mw(train)
    binary_vectors_train = lsh.get_binary_vector(mw_train, train)
    sig_matrix_train = lsh.min_hash(binary_vectors_train, fraction=params['fraction'])
    
    n_train = sig_matrix_train.shape[0]
    factors_train = get_factors(n_train)
    
    valid_factors = [f for f in factors_train 
                     if params['threshold_min'] <= ((1.0 / f) ** (1.0 / (n_train // f))) <= params['threshold_max']]
    b_to_try = valid_factors if len(valid_factors) <= 3 else [
        valid_factors[0], valid_factors[len(valid_factors) // 2], valid_factors[-1]
    ]
    
    keys_train = list(train.keys())
    best_params, best_f1 = {}, 0
    
    print(f"Tuning parameters across {len(b_to_try)} b values: {b_to_try}...")
    
    for b in b_to_try:
        candidate_pairs = lsh.lsh(sig_matrix_train, keys_train, b)
        for gamma in params['gammas']:
            for epsilon in params['epsilons']:
                for mu in params['mus']:
                    clustered, dissimilarity = msm.main(candidate_pairs, train, gamma, epsilon, mu)
                    f1 = check_duplicates(train, dissimilarity, candidate_pairs, clustered)[0]
                    if f1 > best_f1:
                        best_params = {'gamma': gamma, 'epsilon': epsilon, 'mu': mu}
                        best_f1 = f1
    
    print(f"Best params: {best_params}, F1: {best_f1:.4f}\n")
    
    mw_test = lsh.get_mw(test)
    binary_vectors_test = lsh.get_binary_vector(mw_test, test)
    sig_matrix_test = lsh.min_hash(binary_vectors_test, fraction=params['fraction'])
    
    n_test = sig_matrix_test.shape[0]
    factors_test = [f for f in get_factors(n_test) if f >= 15]
    
    columns = ["Threshold", "F1", "F1*", "PC", "PQ", "Number Comparisons", "Fraction Comparisons", "Precision", "Recall"]
    store_df = pd.DataFrame(np.zeros((len(factors_test), len(columns))), columns=columns, index=factors_test)
    
    keys_test = list(test.keys())
    
    for factor in factors_test:
        candidate_pairs = lsh.lsh(sig_matrix_test, keys_test, factor)
        r = n_test // factor
        threshold = (1.0 / factor) ** (1.0 / r)
        clustered, dissimilarity = msm.main(candidate_pairs, test, best_params['gamma'], best_params['epsilon'], best_params['mu'])
        store_df.loc[factor] = [threshold] + list(check_duplicates(test, dissimilarity, candidate_pairs, clustered))
    
    metrics_summary = calculate_metrics_summary(store_df, best_params)
    print_metrics_summary(metrics_summary, bootstrap_id)
    
    save_checkpoint(checkpoint_dir, bootstrap_id, store_df, metrics_summary)
    
    return store_df, metrics_summary

def run_single_experiment(path, bootstraps, checkpoint_dir, params, data_correction):
    """Run a single experiment with given data_correction setting"""
    params['data_correction'] = data_correction
    
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    completed_bootstraps = get_completed_bootstraps(checkpoint_dir)
    remaining_bootstraps = [i for i in range(bootstraps) if i not in completed_bootstraps]
    
    print(f"Found {len(completed_bootstraps)} completed bootstraps")
    print(f"Need to run {len(remaining_bootstraps)} remaining bootstraps\n")
    
    results = {}
    metrics_summaries = {}
    
    for bootstrap_id in completed_bootstraps:
        checkpoint = load_checkpoint(checkpoint_dir, bootstrap_id)
        if checkpoint:
            results[bootstrap_id] = checkpoint['results']
            metrics_summaries[bootstrap_id] = checkpoint['metrics_summary']
            print(f"Loaded bootstrap {bootstrap_id + 1} from checkpoint")
    
    if remaining_bootstraps:
        bootstrap_args = [(data, i, checkpoint_dir, params) for i in remaining_bootstraps]
        num_workers = min(len(remaining_bootstraps), cpu_count())
        print(f"Running {len(remaining_bootstraps)} bootstraps in parallel using {num_workers} workers\n")
        
        with Pool(num_workers) as pool:
            bootstrap_results = pool.map(run_bootstrap, bootstrap_args)
        
        for i, (store_df, metrics_summary) in zip(remaining_bootstraps, bootstrap_results):
            results[i] = store_df
            metrics_summaries[i] = metrics_summary
    
    results_list = [results[i] for i in range(bootstraps)]
    average_df = sum(results_list) / len(results_list)
    
    return average_df, metrics_summaries

def main_func(path="TVs-all-merged.json", path_res=".", bootstraps=10, checkpoint_dir="checkpoints", 
              gammas=[0.75], epsilons=[0.5], mus=[0.7], fraction=0.5, threshold_min=0.15, threshold_max=0.4,
              data_correction=True, compare=False):
    
    params = {
        'gammas': gammas,
        'epsilons': epsilons,
        'mus': mus,
        'fraction': fraction,
        'threshold_min': threshold_min,
        'threshold_max': threshold_max
    }
    
    if compare:
        print("="*60)
        print("RUNNING COMPARISON: WITH vs WITHOUT DATA CORRECTION")
        print("="*60)
        
        checkpoint_dir_corrected = checkpoint_dir + "_corrected"
        checkpoint_dir_no_correction = checkpoint_dir + "_no_correction"
        
        print("\n--- Running WITH data correction ---")
        average_df_corrected, metrics_summaries_corrected = run_single_experiment(
            path, bootstraps, checkpoint_dir_corrected, params.copy(), True
        )
        
        print("\n--- Running WITHOUT data correction ---")
        average_df_no_correction, metrics_summaries_no_correction = run_single_experiment(
            path, bootstraps, checkpoint_dir_no_correction, params.copy(), False
        )
        
        print("\n" + "="*60)
        print("FINAL RESULTS COMPARISON")
        print("="*60)
        print("\nWITH Data Correction:")
        print(average_df_corrected)
        print("\nWITHOUT Data Correction:")
        print(average_df_no_correction)
        
        final_metrics_corrected = calculate_metrics_summary(average_df_corrected)
        final_metrics_no_correction = calculate_metrics_summary(average_df_no_correction)
        
        print("\n--- Metrics WITH Data Correction ---")
        print_metrics_summary(final_metrics_corrected)
        print("\n--- Metrics WITHOUT Data Correction ---")
        print_metrics_summary(final_metrics_no_correction)
        
        plot_metrics(average_df_corrected, average_df_no_correction, "Fraction Comparisons", "F1", path_res)
        plot_metrics(average_df_corrected, average_df_no_correction, "Fraction Comparisons", "F1*", path_res)
        plot_metrics(average_df_corrected, average_df_no_correction, "Fraction Comparisons", "PC", path_res)
        
        pq_corrected = average_df_corrected[average_df_corrected['Fraction Comparisons'] > 0]
        pq_no_correction = average_df_no_correction[average_df_no_correction['Fraction Comparisons'] > 0]
        plot_metrics(pq_corrected, pq_no_correction, "Fraction Comparisons", "PQ", path_res)
        
        plot_metrics(average_df_corrected, average_df_no_correction, "Fraction Comparisons", "Precision", path_res)
        plot_metrics(average_df_corrected, average_df_no_correction, "Fraction Comparisons", "Recall", path_res)
        
        return average_df_corrected, average_df_no_correction
    else:
        average_df, metrics_summaries = run_single_experiment(path, bootstraps, checkpoint_dir, params, data_correction)
        
        final_results_file = os.path.join(checkpoint_dir, "final_results.pkl")
        with open(final_results_file, 'wb') as f:
            pickle.dump(average_df, f)
        print(f"Saved final results to {final_results_file}\n")
        
        correction_label = "WITH Data Correction" if data_correction else "WITHOUT Data Correction"
        print(f"\nFinal Results ({correction_label}):")
        print(average_df)
        
        final_metrics = calculate_metrics_summary(average_df)
        
        if metrics_summaries:
            gammas = [m['best_gamma'] for m in metrics_summaries.values() if 'best_gamma' in m]
            epsilons = [m['best_epsilon'] for m in metrics_summaries.values() if 'best_epsilon' in m]
            mus = [m['best_mu'] for m in metrics_summaries.values() if 'best_mu' in m]
            if gammas:
                final_metrics.update({
                    'avg_gamma': np.mean(gammas), 'std_gamma': np.std(gammas),
                    'avg_epsilon': np.mean(epsilons), 'std_epsilon': np.std(epsilons),
                    'avg_mu': np.mean(mus), 'std_mu': np.std(mus)
                })
        
        print_metrics_summary(final_metrics)
        
        if 'avg_gamma' in final_metrics:
            print(f"\n{'='*60}")
            print("PARAMETER STATISTICS ACROSS BOOTSTRAPS")
            print(f"{'='*60}")
            print(f"Gamma:    avg={final_metrics['avg_gamma']:.4f}, std={final_metrics['std_gamma']:.4f}")
            print(f"Epsilon:  avg={final_metrics['avg_epsilon']:.4f}, std={final_metrics['std_epsilon']:.4f}")
            print(f"Mu:       avg={final_metrics['avg_mu']:.4f}, std={final_metrics['std_mu']:.4f}")
            print(f"{'='*60}\n")
        
        return average_df

if __name__ == "__main__":
    main_func(
        path="TVs-all-merged.json",
        path_res=".",
        bootstraps=8,
        checkpoint_dir="checkpoints",
        gammas=[0.5, 0.6, 0.7, 0.75],
        epsilons=[0.4, 0.5, 0.6],
        mus=[0.6, 0.65, 0.7],
        fraction=0.5,
        threshold_min=0.15,
        threshold_max=0.4,
        compare=True
    )
