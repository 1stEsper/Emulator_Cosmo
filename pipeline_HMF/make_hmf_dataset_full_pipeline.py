import numpy as np
import pandas as pd
from scipy.stats import qmc
from symbolic_pofk.pk_to_hmf import full_pipeline_hmf  

# ===== CONFIG =====
n_cosmo = 2000
n_mass = 40
n_z = 12

z_vals = 0
M_vals = np.logspace(np.log10(1e12),np.log10(5e15), n_mass)

# 7 cosmological + 4 baryonic parameters
param_ranges = {
    "As":    (2.7, 3.2),   # 2.5 to 3.5
    "Om":    (0.01, 0.6),
    "Ob":    (0.001, 0.1),
    "h":     (0.5, 1.0),
    "ns":    (0.92, 1.04),
    "w0":    (-1.3, -0.7),
    "wa":    (-0.5, 0.5),
    "A_SN1": (1e-5, 5.0),
    "A_SN2": (1e-5, 5.0),
    "A_AGN1":(1e-5, 5.0),
    "A_AGN2":(1e-5, 5.0)
}

# ===== LATIN HYPERCUBE SAMPLING (Maximin) =====
d = len(param_ranges)
best_sample = None
best_min_dist = -np.inf

for _ in range(50):
    sampler = qmc.LatinHypercube(d=d)
    sample = sampler.random(n_cosmo)
    dist_matrix = np.sqrt(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = dist_matrix.min()
    if min_dist > best_min_dist:
        best_min_dist = min_dist
        best_sample = sample

scaled_sample = qmc.scale(best_sample,
                          [v[0] for v in param_ranges.values()],
                          [v[1] for v in param_ranges.values()])

param_keys = list(param_ranges.keys())
params = {key: scaled_sample[:, i] for i, key in enumerate(param_keys)}

# ===== COMPUTE HMFs WITH PIPELINE =====
rows = []

for i in range(n_cosmo):
    if params["Ob"][i] >= params["Om"][i]:
        continue
        
    # for z in z_vals:
    #     try:
    #         hmf_vals, sigma_vals, sigma8 = full_pipeline_hmf(
    #             As=params["As"][i], Om=params["Om"][i], Ob=params["Ob"][i],
    #             h=params["h"][i], ns=params["ns"][i],
    #             w0=params["w0"][i], wa=params["wa"][i],
    #             A_SN1=params["A_SN1"][i], A_SN2=params["A_SN2"][i],
    #             A_AGN1=params["A_AGN1"][i], A_AGN2=params["A_AGN2"][i],
    #             z=z, M_vals=M_vals, baryon_effect=True
    #         )

    #         for j, M in enumerate(M_vals):
    #             rows.append([
    #                 i, params["As"][i], params["Om"][i], params["Ob"][i], params["h"][i], params["ns"][i],
    #                 params["w0"][i], params["wa"][i],
    #                 params["A_SN1"][i], params["A_SN2"][i],
    #                 params["A_AGN1"][i], params["A_AGN2"][i],
    #                 sigma8, M, z, sigma_vals[j], hmf_vals[j]
    #             ])
    #     except Exception as e:
    #         print(f"Skipping cosmo {i}, z={z}: {str(e)}")
    #         continue
    try:
        hmf_vals, sigma_vals, sigma8 = full_pipeline_hmf(
            As=params["As"][i], Om=params["Om"][i], Ob=params["Ob"][i],
            h=params["h"][i], ns=params["ns"][i],
            # w0=params["w0"][i], wa=params["wa"][i],
            w0 = -1.0, wa=0.0,
            A_SN1=params["A_SN1"][i], A_SN2=params["A_SN2"][i],
            A_AGN1=params["A_AGN1"][i], A_AGN2=params["A_AGN2"][i],
            z=z_vals, M_vals=M_vals, baryon_effect=False
        )

        for j, M in enumerate(M_vals):
            rows.append([
                i, params["As"][i], params["Om"][i], params["Ob"][i], params["h"][i], params["ns"][i],
                # params["w0"][i], params["wa"][i],
                # params["A_SN1"][i], params["A_SN2"][i],
                # params["A_AGN1"][i], params["A_AGN2"][i],
                sigma8, M, z_vals, sigma_vals[j], hmf_vals[j]
            ])
    except Exception as e:
        print(f"Skipping cosmo {i}, z={z_vals}: {str(e)}")
        continue
    

# ===== SAVE TO CSV =====
df = pd.DataFrame(rows, columns=[
    'Cosmo_ID', 'As', 'Om', 'Ob', 'h', 'ns', 
    # 'w0', 'wa',
    # 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2',
    'sigma8', 'Mass', 'Z', 'Sigma', 'HMF'
])
df.to_csv("HMF_LCDM_noBaryon_2000cp_40Mz0.csv", index=False)
print(f"✅ Done! Saved {len(df)} rows to HMF_dataset_with_baryons.csv")
