import logging
import pandas as pd
import numpy as np
from typing import Dict
 
logger = logging.getLogger(__name__)

def binning(datasets : Dict[str, pd.DataFrame], buckets, features):
    logger.info("Binning")
    df_dict = {}
    for host, dataset in datasets.items():
        df = pd.DataFrame()
        for feature in dataset.columns:
            if feature in features:
                df[feature] = pd.cut(dataset[feature], bins=(np.linspace(0, 1, buckets+1)), right=True, include_lowest=True, labels=False)
            else:
                df[feature] = dataset[feature].copy()
        df_dict[host] = df
    logger.info("Binning complete")
    return df_dict