import os
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
import logging

logger = logging.getLogger(__name__)


def plot_feature_over_time(datasets : Dict[str, pd.DataFrame], numerical_features):
    datasetsOrderedByTime = {host : dataset.sort_values(by="FLOW_START_MILLISECONDS")  for host, dataset in datasets.items()}

    for feature in numerical_features:
        for host1, dataset1 in datasetsOrderedByTime.items():
            fpath = f"graphs/hostsProva/{host1}/numerical/compare/{feature}"
            if not os.path.exists(fpath):
                os.makedirs(name=fpath, exist_ok=True)
            for host2, dataset2 in datasetsOrderedByTime.items():
                if(host1 == host2):
                    pass
                else:
                    plt.figure(figsize=(30,10))
                    plt.plot(dataset1[feature], label=f"{host1}", color="green")
                    plt.plot(dataset2[feature], label=f"{host2}", color="red")
                    plt.fill_between(range(len(dataset1)), dataset1[feature], alpha=0.2)
                    plt.fill_between(range(len(dataset2)), dataset2[feature], alpha=0.2)
                    plt.legend()
                    plt.grid()
                    plt.xlabel("Time")
                    plt.ylabel("Value")
                    plt.title(label=f"{feature}", pad=20)
                    filename = os.path.join(fpath, f"{host1}vs{host2}.jpg")
                    plt.savefig(filename)
                    plt.close()
            print(f"Saved all images regarding {host1} in {fpath}")
        

def compare_distributions(datasets : Dict[str, pd.DataFrame], numerical_features):
    # Per ogni feature definiscono le distribuzioni per dataset
    for feature in numerical_features:
        logger.info("Next feature")
        distributions = {}
        for host, dataset in datasets.items():
            kde = gaussian_kde(dataset[feature])  # Non conosco la distribuzione, faccio kde
            pdf = kde(np.linspace(0, 1, 1000)) # Prendo 1000 sample per definire la distribuzione
            pdf += 1e-10
            pdf /= np.sum(pdf) # Normalizzo per garantire che la somma dell'area sotto la curva sia 1
            distributions[host] = pdf

        logger.info("Distributions computed")
        cmp_dict= {}
        for host1 in datasets.keys():
            cmp = {} 
            for host2 in datasets.keys():
                if(host1 == host2):
                    cmp[host2] = 0 # Due approssimazioni uguali hanno distanza 0
                else:
                    # Kullback Leibler per definire quanto male dist host2 approssima dist host1
                    cmp[host2] = entropy(distributions[host1], distributions[host2]) 
                logger.info(f"Compared{host1} and {host2}")
            cmp_dict[host1] = pd.Series(cmp) 
        cmp_df = pd.DataFrame(cmp_dict).T
        print(f"\nFeature: {feature}")
        print(cmp_df, end="\n")

def compare_distributions2(datasets : Dict[str, pd.DataFrame], feature_names, attack = None):
    print(feature_names)

    fpath = f"resources/csvs"
    if(attack != None):
        fpath = os.path.join(fpath, f"attacks/{attack}")
        print("---------------------------------------------------------")
        print(f"                       {attack}")
        print("---------------------------------------------------------")
    else:
        fpath = os.path.join(fpath, "normal")

    for feature in feature_names:
        dist_prob_dict = {} # Qui finiscono tutte le distribuzioni di probabilità
        newfpath = os.path.join(fpath, f"{feature}")
         
        if not os.path.exists(newfpath):
            os.makedirs(name=newfpath, exist_ok=True)

        for host, dataset in datasets.items():
            prob = round((dataset[feature].value_counts(normalize=True) * 100), 3)
            dist_prob_dict[host] = prob

        # Lista di liste di tutti i valori della feature tra tutti i dataset
        values = [dist_prob.index for dist_prob in dist_prob_dict.values()]

        # Tutti i valori in ordine decrescente
        all_values = sorted(set().union(*values))

        # df con tutte le probabilità
        prob_df = pd.DataFrame(index=all_values)
        for host, dist_prob in dist_prob_dict.items():
            prob_df[host] = dist_prob

        # Ci sono dataset che non hanno alcuni valori, si assume probabilità 0
        prob_df = prob_df.fillna(0)
                    
        print(f"Feature: {feature}", end="\n")
        # print("Probability of occurrency")
        # print(prob_df, end="\n")
        
        for host1 in prob_df.columns:
            differences_df = pd.DataFrame(index=["std", "mean", "sum"])
            for host2 in prob_df.columns:
                if(host1 != host2):
                    difference = abs(prob_df[host1] - prob_df[host2])
                    difference_infos = {"sum" : difference.sum(),
                                        "std" : difference.std(),
                                        "mean" : difference.mean()}
                    differences_df[host2] = difference_infos

            filename = os.path.join(newfpath, f"{host1}.csv")
            differences_df.to_csv(filename)
            print("------------------------------------------------------------------------------------")
            print(f"Difference infos between {host1} and the others: ")
            print(differences_df.T)
            print("------------------------------------------------------------------------------------")
        
        print("\n\n\n\n\n\n\n\n")


def plot_feature_prob_dist(datasets : Dict[str, pd.DataFrame], numerical_features, attack = None):
    for feature in numerical_features:
        if(attack != None):
            fpath = f"graphs/hostsProva/dist/{attack}"
        else:                
            fpath = "graphs/hostsProva/dist"
        if not os.path.exists(fpath):
            os.makedirs(name=fpath, exist_ok=True)

        i = 1
        plt.figure(figsize=(15,30))
        plt.title(label=f"Probability Distribution: {feature}", pad=30)
        plt.subplots_adjust(hspace=0.5)
        plt.gca().set_axis_off()
        for host, dataset in datasets.items():
            prob = dataset[feature].value_counts(normalize=True).sort_index()
            values = prob.index

            plt.subplot(5, 1, i)
            plt.plot(values, prob, color="orange")
            plt.fill_between(values, prob, color="orange", alpha=0.2)
            plt.xlabel("Value")
            plt.ylabel("Frequency/Instances")
            plt.title(f"Host: {host}", fontsize=12, pad=10)
            plt.grid()
            i += 1

        filename = os.path.join(fpath, f"{feature}.jpg")
        plt.savefig(filename)
        plt.close()
    print(f"All figures saved in {fpath}")

def compare_plots(datasets : Dict[str, pd.DataFrame], numerical_features : List[str], attack = None):
    numerical_features = numerical_features.copy()
    numerical_features.remove("FLOW_START_MILLISECONDS")
    numerical_features.remove("FLOW_END_MILLISECONDS")
    for feature in numerical_features:
        if(attack != None):
            fpath = f"graphs/hostsProva/dist/{attack}/compare/{feature}"
        else:                
            fpath = f"graphs/hostsProva/dist/compare/{feature}"
        if not os.path.exists(fpath):
            os.makedirs(name=fpath, exist_ok=True)

        for host1, dataset1 in datasets.items():
            prob1 = dataset1[feature].value_counts(normalize=True).sort_index()

            for host2, dataset2 in datasets.items():
                if(host1 != host2):
                    prob2 = dataset2[feature].value_counts(normalize=True).sort_index()

                    common_values = sorted(set(prob1.index).union(set(prob2.index)))

                    prob1_aligned = prob1.reindex(common_values, fill_value=0)
                    prob2_aligned = prob2.reindex(common_values, fill_value=0)
                   
                    y1,y2 = prob1_aligned.values, prob2_aligned.values

                    plt.figure(figsize=(30, 10))
                    plt.title(label=f"Probability Distribution: {feature}", pad=20)
                    plt.plot(common_values, y1, label=f"{host1}", color="orange")
                    plt.plot(common_values, y2, label=f"{host2}", color="green")
                    plt.fill_between(common_values, y1, color="orange", alpha=0.2)
                    plt.fill_between(common_values, y2, color="green", alpha=0.2)

                    plt.fill_between(common_values, y1, y2, where=(y1 > y2), interpolate=True, label="host1 > host2",
                                    color=(127/255, 210/255, 10/255))

                    plt.fill_between(common_values, y1, y2, where=(y1 < y2), interpolate=True, label="host1 < host2",
                                    color=(210/255, 127/255, 10/255))

                    plt.xlabel("Value")
                    plt.ylabel("Frequency / Instances")
                    plt.legend()
                    plt.grid()
                    
                    filename = os.path.join(fpath, f"{host1}vs{host2}.jpg")
                    plt.savefig(filename)
                    plt.close()
                        

def binning(datasets : Dict[str, pd.DataFrame], buckets):
    logger.info("Binning")
    bin_labels = list(range(buckets))
    df_dict = {}
    for host, dataset in datasets.items():
        df = pd.DataFrame()
        for feature in dataset.columns:
            df[feature] = pd.cut(dataset[feature], bins=(np.linspace(0, 1, buckets+1)), right=True, include_lowest=True, labels=bin_labels)
        df_dict[host] = df
    logger.info("Done")
    return df_dict