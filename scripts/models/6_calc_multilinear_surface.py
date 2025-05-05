#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch

from scripts.models.hierarchical_model import HierarchicalModel


def get_phi(vmin=0.01, vmax=25, n=25):
    return np.log(np.geomspace(vmin, vmax, n))


def calc_mu(phi1, phi2, model):
    wt = model.theta_wt.detach().item()
    xs_, ys_ = torch.Tensor(phi1 - wt), torch.Tensor(phi2 - wt)
    zs = model.calc_multilinear_function(xs_, ys_).detach().numpy()
    return zs


if __name__ == "__main__":
    gt_data = pd.read_csv("results/genotype_predictions.csv", index_col=0)
    model_pred = gt_data["hierarchical_pred"].to_dict()
    model = torch.load("results/hierarchical.pkl")

    # Calculate multilinear surface
    phi1 = get_phi()
    phi2 = get_phi()
    xs, ys = np.meshgrid(phi1, phi2)
    zs = calc_mu(xs, ys, model)

    np.save("results/multilinear.xs.npy", xs)
    np.save("results/multilinear.ys.npy", ys)
    np.save("results/multilinear.zs.npy", zs)

    # Calculate multilinear dense surface for heatmap
    phi1 = get_phi(vmax=30, n=100)
    phi2 = get_phi(vmax=30, n=100)
    xs, ys = np.meshgrid(phi1, phi2)
    zs = calc_mu(xs, ys, model)

    np.save("results/multilinear.dense.xs.npy", xs)
    np.save("results/multilinear.dense.ys.npy", ys)
    np.save("results/multilinear.dense.zs.npy", zs)

    # Calculate transects
    plt_transects = {}
    gts = {
        "WW": ("W_W_W_W_Summer 22", "W_W_M_M8_Summer 23"),
        "MW": ("M_W_W_W_Summer 22", "W_W_M_W_Summer 23"),
        "MH": ("M_H_W_W_Summer 22", "W_W_M_M8_Summer 23"),
    }

    n = 100
    for label, (gt, gt_max) in gts.items():
        phi2 = np.full(n, fill_value=model_pred[gt])

        phi1 = get_phi(n=n)
        mu = calc_mu(phi1, phi2, model)
        plt_transects[f"phi_sep_{label}"] = phi1
        plt_transects[f"phi_plt_{label}"] = phi2
        plt_transects[f"mu_{label}"] = mu

        phi1 = get_phi(vmax=np.exp(model_pred[gt_max]), n=n)
        mu = calc_mu(phi1, phi2, model)
        plt_transects[f"segment_phi_sep_{label}"] = phi1
        plt_transects[f"segment_phi_plt_{label}"] = phi2
        plt_transects[f"segment_mu_{label}"] = mu

    plt_transects = pd.DataFrame(plt_transects)
    plt_transects["target"] = "PLT"

    sep_transects = {}
    gts = {
        "WW": ("W_W_W_W_Summer 23", "M_H_W_W_Summer 22"),
        "MW": ("W_W_M_W_Summer 23", "M_W_W_W_Summer 22"),
        "MM": ("W_W_M_M8_Summer 23", "M_H_W_W_Summer 22"),
    }

    for label, (gt, gt_max) in gts.items():
        phi1 = np.full(100, fill_value=model_pred[gt])

        phi2 = get_phi(n=n)
        mu = calc_mu(phi1, phi2, model)
        sep_transects[f"phi_sep_{label}"] = phi1
        sep_transects[f"phi_plt_{label}"] = phi2
        sep_transects[f"mu_{label}"] = mu

        phi2 = get_phi(vmax=np.exp(model_pred[gt_max]), n=n)
        mu = calc_mu(phi1, phi2, model)
        sep_transects[f"segment_phi_sep_{label}"] = phi1
        sep_transects[f"segment_phi_plt_{label}"] = phi2
        sep_transects[f"segment_mu_{label}"] = mu

    sep_transects = pd.DataFrame(sep_transects)
    sep_transects["target"] = "SEP"

    transects = pd.concat([plt_transects, sep_transects], axis=0)
    transects.to_csv("results/multilinear.transects.csv")
