import logging
import matplotlib.pyplot as plt
import pandas as pd 

divided = 0
normalized = 0
logger = logging.getLogger(__name__)

# Funzione per vedere eterogeneità delle label fra i dataset
def labels_study(datasets) -> None:
    # Raggruppo dataset per host e conto il numero di tipologie di attacco
    logger.info("Grouping dataset per host..")
    df = pd.DataFrame()
    for host, dataset in datasets.items(): 
        df_melted = dataset.melt(id_vars=dataset.columns.drop(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]),value_vars=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
        df_filtered = df_melted.loc[df_melted["value"].isin([host])]
        df = pd.concat([df_filtered, df])

    #Raggruppo dataset per ip
    df_grouped_by_ip = df.groupby("value")

    type_of_attacks_per_host = df_grouped_by_ip["Attack"].value_counts().unstack(fill_value=0).sort_values(by="Benign", ascending=False)
    logger.info("Dataset grouped correctly")

    print(type_of_attacks_per_host)

    # Creazione grafici per facilitare comparazione degli attacchi tra host
    i = 1
    for attack_type in type_of_attacks_per_host.columns:
        plt.figure(10)
        plt.subplot(3, 3, i)
        i += 1
        type_of_attacks_per_host[attack_type].plot(kind="bar", xlabel="host", ylabel="number of attack rows", title=str(attack_type), figsize=(15, 8))
    plt.tight_layout()
    plt.show()    