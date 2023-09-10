# %%
import numpy as np
import pandas as pd

from IPython.display import display

pd.set_option("display.max_columns", None)

# %%
df_raw = (
    pd.read_csv("../../data/raw/trajectory_data.csv")
    .sort_values(["HPCID", "SM_DATE"])
    .reset_index(drop=True)
    .assign(
        sex = lambda df: np.where(df["sex"] == True, 1, 0),
        MVPA = lambda df: np.where(df["MVPA"] == True, 1, 0),
        Smoke = lambda df: np.where(df["Smoke"] == True, 1, 0),
        Diabetes = lambda df: np.where(df["Diabetes"] == True, 1, 0),
        Hypertension = lambda df: np.where(df["Hypertension"] == True, 1, 0),
        HTN_med = lambda df: np.where(df["HTN_med"] == True, 1, 0),
        Hyperlipidemia = lambda df: np.where(df["Hyperlipidemia"] == True, 1, 0),
        Hepatatis = lambda df: np.where(df["Hepatatis"] == True, 1, 0),
        ALC = lambda df: np.where(df["ALC"] == True, 1, 0),
        MED_HYPERTENSION = lambda df: np.where(df["MED_HYPERTENSION"] == True, 1, 0),
        MED_HYPERLIPIDEMIA = lambda df: np.where(df["MED_HYPERLIPIDEMIA"] == True, 1, 0),
    )
    .drop(
        columns=["CDW_NO", "ID"]
    )
)

display(df_raw.head())
# %%

# Inclusion Exclusion Criteria


## RER >= 1.1 and baPWV missing excluded

### Inclusion (M: 25738, F: 6131)
### RER >= 1.1 (M: 11835, F: 1871)
### baPWV not missed (M: 3924, F: 1011)


df_selected = (
    df_raw
    .query(
        "RER_over_gs == True"
    )
    .pipe(
        lambda df: print(
            df
            .groupby(["HPCID"])
            .head(1)
            ["sex"].value_counts(dropna=False)
        ) or df
    )
    .query(
        "baPWV.notnull()"
    )
    .pipe(
        lambda df: print(
            df
            .groupby(["HPCID"])
            .head(1)
            ["sex"].value_counts(dropna=False)
        ) or display(df) or df
    )
)

# %%
df_preprocessed = (
    df_selected
    .assign(
        R_Cholesterol_tmp = lambda df: df["CHOLESTEROL"] - df["HDL_C"] - df["LDL_C"],
        R_Cholesterol = lambda df: np.where(df["R_Cholesterol_tmp"] < 0, np.nan, df["R_Cholesterol_tmp"]),
        non_HDL_C = lambda df: df["CHOLESTEROL"] - df["HDL_C"],
    )
    .drop(
        columns = ["R_Cholesterol_tmp"]
    )
)


# %%

def make_age_sex_adjusted_crf_tertile(
    dataframe:pd.DataFrame,
):
    
    df_male = dataframe.query("sex == False").reset_index(drop=True)
    df_female = dataframe.query("sex == True").reset_index(drop=True)
    
    
    df_male = (
        df_male
        .assign(
            age_group = lambda df: np.where(
                df["AGE"] < 30,
                "male_20_30",
                np.where(
                    df["AGE"] < 40,
                    "male_30_40",
                    np.where(
                        df["AGE"] < 50, 
                        "male_40_50",
                        np.where(
                            df["AGE"] < 60,
                            "male_50_60",
                            np.where(
                                df["AGE"] < 70,
                                "male_60_70",
                                "male_70_83"
                            )
                        )
                    )
                )
            )
        )
    )
    
    df_female = (
        df_female
        .assign(
            age_group = lambda df: np.where(
                df["AGE"] < 40,
                "female_30_40",
                np.where(
                    df["AGE"] < 50,
                    "female_40_50",
                    np.where(
                        df["AGE"] < 60, 
                        "female_50_60",
                        "female_60_73",
                    )
                )
            )
        )
    )
    
    df_tertile = (
        pd.concat(
            [
                df_male, df_female
            ],
            axis=0,
            ignore_index=True,
        )
        .assign(
            CRF_tertile = lambda df: (
                df
                .groupby(["age_group"])
                ["CRF"]
                .transform(
                    lambda seris: pd.qcut(
                        seris, 
                        q=3, 
                        labels=["T1", "T2", "T3"]
                    )
                )
            )
        )
    )
    
    
    return df_tertile
# %%

df_preprocessed = make_age_sex_adjusted_crf_tertile(df_preprocessed)

(
    df_preprocessed
    .to_csv("../../data/preprocessed/stat_set.csv", index=False)
)

(
    df_preprocessed
    .fillna(df_preprocessed.median())
    .to_csv("../../data/preprocessed/modeling_set.csv", index=False)
)


# %%
