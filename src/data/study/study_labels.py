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
        print(host)
        print(dataset["Attack"].value_counts())