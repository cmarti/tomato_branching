#!/usr/bin/env python
import pandas as pd

from scripts.settings import EJ2_SERIES


def relabel_ej2(x):
    seq = ["W"] * 6
    if x != "W":
        a, p = x[0], EJ2_SERIES.index(x[1])
        seq[p] = a
    return "".join(seq)


if __name__ == "__main__":
    # Load raw data
    print("Processing raw data for modeling")
    data = pd.read_csv("data/Branching_Master.csv")
    data["plant_id"] = [
        "{}.{}".format(a, b) for a, b in zip(data["Pedigree"], data["Plant"])
    ]
    cols = [
        "Season",
        "Pedigree",
        "Name",
        "EJ2 Allele",
        "PLT380",
        "PLT710",
        "J2",
        "EJ2 Genotype",
        "EJ2",
        "Genotype",
        "Specific Genotype",
        "Plant",
        "plant_id",
    ]

    # Get one raw per plant
    data = pd.melt(data, id_vars=cols).dropna()

    # Remove not quantitative phenotypes
    data = data.loc[data["value"] != "Proliferated", :]
    data = data.loc[data["value"] != "proliferated", :]
    data = data.loc[data["value"] != "inhibited", :]
    data = data.loc[data["value"] != "**", :]
    data["value"] = data["value"].astype(int)
    data.loc[data["value"] > 60, "value"] = 60
    data = data.loc[data["value"] >= 0, :]

    # Select relevant columns
    cols = [
        "Season",
        "Pedigree",
        "Specific Genotype",
        "EJ2 Allele",
        "EJ2",
        "J2",
        "PLT380",
        "PLT710",
        "plant_id",
        "variable",
        "value",
    ]
    data = data[cols]

    # Select relevant genotypes
    data = data.loc[data["EJ2"] != "M2", :]
    data = data.loc[data["EJ2"] != "Me", :]
    data = data.loc[data["EJ2"] != "He", :]
    data = data.loc[data["EJ2"] != "Ms", :]
    data = data.loc[data["EJ2 Allele"] != "e", :]

    # Summarize per plant
    plant_data = (
        data.groupby(
            [
                "plant_id",
                "EJ2 Allele",
                "EJ2",
                "J2",
                "PLT380",
                "PLT710",
                "Season",
            ]
        )
        .agg({"value": ("sum", "count", "mean", "var")})["value"]
        .reset_index()
    )
    plant_data.columns = [
        "plant_id",
        "EJ2 variant",
        "EJ2",
        "J2",
        "PLT3",
        "PLT7",
        "Season",
        "branches",
        "influorescences",
        "obs_mean",
        "variance",
    ]

    # Save output
    plant_data.to_csv("data/plant_data.csv")

    # Summarize per genotype
    gt_data = (
        data.groupby(
            [
                "EJ2",
                "J2",
                "PLT380",
                "PLT710",
                "Season",
            ]
        )
        .agg({"value": ("mean", "var")})["value"]
        .reset_index()
    )
    gt_data.columns = [
        "EJ2",
        "J2",
        "PLT3",
        "PLT7",
        "Season",
        "obs_mean",
        "variance",
    ]
    gt_data.to_csv("data/gt_data.csv")
