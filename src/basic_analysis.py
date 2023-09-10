# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
pd.set_option("display.max_columns", None)
# %%
df_orig = (
    pd.read_csv(
        "../data/trajectory_data.csv", 
    )
    .assign(
        SM_DATE = lambda x: x['SM_DATE'].astype("datetime64"), 
        last_visit = lambda x: x.groupby(['HPCID'])['SM_DATE'].transform(max), 
        follow_up = lambda x: (x['last_visit'] - x['SM_DATE']) / np.timedelta64(1, 'M')
    )
)

# %%
df_bapwv = df_orig.query("baPWV.notnull()", engine='python').reset_index(drop=True)

display(df_bapwv["HPCID"].value_counts().value_counts())
# %%
df_abi = df_orig.query("ABI.notnull()", engine='python').reset_index(drop=True)

display(df_abi["HPCID"].value_counts().value_counts())

# %%
df_imt = df_orig.query("mean_IMT.notnull()", engine='python').reset_index(drop=True)

display(df_imt["HPCID"].value_counts().value_counts())
# %%
