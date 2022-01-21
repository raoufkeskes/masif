import sys
import os
import numpy as np
from IPython.core.debugger import set_trace
import importlib
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from default_config.masif_opts import masif_opts
import pickle
from masif_modules.protein import Protein
from sklearn.neighbors import KDTree





"""
masif_ppi_search_cache_training_data.py: Function to cache all the training data for MaSIF-search. 
                This function extract all the positive pairs and a random number of negative surfaces.
                In the future, the number of negative surfaces should be increased.
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

params = masif_opts['ppi_search']

if len(sys.argv) > 0:
    custom_params_file = sys.argv[1]
    custom_params = importlib.import_module(custom_params_file, package=None)
    custom_params = custom_params.custom_params

    for key in custom_params:
        print('Setting {} to {} '.format(key, custom_params[key]))
        params[key] = custom_params[key]

if 'pids' not in params:
    params['pids'] = ['p1', 'p2']

# Read the positive first
parent_in_dir = params['masif_precomputation_dir']

target_rho_wrt_center = []
target_theta_wrt_center = []
target_input_feat = []
target_mask = []

pos_rho_wrt_center = []
pos_theta_wrt_center = []
pos_input_feat = []
pos_mask = []

neg_rho_wrt_center = []
neg_theta_wrt_center = []
neg_input_feat = []
neg_mask = []

np.random.seed(0)
training_idx = []
val_idx = []
test_idx = []
pos_names = []
neg_names = []

# training_list = [x.rstrip() for x in open(params['training_list']).readlines()]
# testing_list = [x.rstrip() for x in open(params['testing_list']).readlines()]

df = pd.read_csv("/home/raoufks/Desktop/Projects/structures_data/tables/antibodies_sample.csv")

df_train = df[df["set"] == "train"]
df_valid = df[df["set"] == "valid"]

df_train_valid = pd.concat((df_train, df_valid))
df_test = df[df["set"] == "test"]

precomp_path = "/media/raoufks/Maxtor/raouf/data/precomputed/"
radius = 12

idx_count = 0
pbar1 = tqdm(iterable=df.iterrows(), total=len(df), desc="training ... ")
for idx, row in pbar1:
    # Read the corresponding ply files.
    pbar1.set_description("Cache data ... " + row["complex_filename"])
    antigen_pkl_path = os.path.join(precomp_path, str(radius) + "A", row["antigen_filename"] + ".pkl")
    antibody_pkl_path = os.path.join(precomp_path, str(radius) + "A", row["antibody_filename"] + ".pkl")

    if os.path.exists(antigen_pkl_path) and os.path.exists(antibody_pkl_path):
        with open(antigen_pkl_path, "rb") as antigen_file, open(antibody_pkl_path, "rb") as antibody_file:
            antigen = Protein(*pickle.load(antigen_file))
            antigen.normalize_hydrophobia()
            antigen.normalize_electrostatics()

            antibody = Protein(*pickle.load(antibody_file))
            antibody.normalize_hydrophobia()
            antibody.normalize_electrostatics()

            # protein.patches_features = protein.patches_features[:, :, [0, 1, 4, 3, 2]]
            iface_labels = antigen.iface["kdtree"]
            if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
            ):
                continue

            # pos_labels: points > max_sc_filt and >  min_sc_filt.
            pos_labels = np.where(iface_labels == 1)[0]
            K = int(params['pos_surf_accept_probability'] * len(pos_labels))
            l = np.arange(len(pos_labels))
            np.random.shuffle(l)
            l = l[:K]
            l = pos_labels[l]

            v1 = antigen.vertices[l]
            v2 = antibody.vertices

            # For each point in v1, find the closest point in v2.
            kdt = KDTree(v2)
            d, r = kdt.query(v1)
            # Contact points: those within a cutoff distance.
            contact_points = np.where(d < params['pos_interface_cutoff'])[0]
            k1 = l[contact_points]
            k2_pos = r[contact_points].reshape(-1)
            # For negatives, get points in v2 far from p1.
            kdt = KDTree(v1)
            dneg, rneg = kdt.query(v2)
            k2_neg = np.where(dneg > params['pos_interface_cutoff'])[0]

            assert len(k1) == len(k2_pos)
            n_pos = len(k1)
            #
            ppi_pair_id = row["complex_filename"]
            pid = row["antigen_filename"]  # target is p1
            for ii in k1:
                pos_names.append('{}_{}_{}'.format(ppi_pair_id, pid, ii))

            target_rho_wrt_center.append(antigen.rhos[k1])
            target_theta_wrt_center.append(antigen.thetas[k1])
            target_input_feat.append(antigen.patches_features[k1])
            target_mask.append(antigen.mask[k1])

            # Read as positives those points.
            pos_rho_wrt_center.append(antibody.rhos[k2_pos])
            pos_theta_wrt_center.append(antibody.thetas[k2_pos])
            pos_input_feat.append(antibody.patches_features[k2_pos])
            pos_mask.append(antibody.mask[k2_pos])
#
            np.random.shuffle(k2_neg)
            k2_neg = k2_neg[:(len(k2_pos))]
            assert(len(k2_neg) == n_pos)
            neg_rho_wrt_center.append(antibody.rhos[k2_neg])
            neg_theta_wrt_center.append(antibody.thetas[k2_neg])
            neg_input_feat.append(antibody.patches_features[k2_neg])
            neg_mask.append(antibody.mask[k2_neg])

            pid = row["antibody_filename"]
            for ii in k2_neg:
                neg_names.append('{}_{}_{}'.format(ppi_pair_id, pid, ii))

            # Training, validation or test?

            if row["set"] == "train":
                training_idx = np.append(training_idx, np.arange(idx_count, idx_count+n_pos))
            elif row["set"] == "valid":
                val_idx = np.append(val_idx, np.arange(idx_count, idx_count+n_pos))
            else:
                test_idx = np.append(test_idx, np.arange(idx_count, idx_count+n_pos))

            idx_count += n_pos
#
if not os.path.exists(params['cache_dir']):
    os.makedirs(params['cache_dir'])
#
target_rho_wrt_center = np.concatenate(target_rho_wrt_center, axis=0)
target_theta_wrt_center = np.concatenate(target_theta_wrt_center, axis=0)
target_input_feat = np.concatenate(target_input_feat, axis=0)
target_mask = np.concatenate(target_mask, axis=0)
#
pos_rho_wrt_center = np.concatenate(pos_rho_wrt_center, axis=0)
pos_theta_wrt_center = np.concatenate(pos_theta_wrt_center, axis=0)
pos_input_feat = np.concatenate(pos_input_feat, axis=0)
pos_mask = np.concatenate(pos_mask, axis=0)
np.save(params['cache_dir']+'/pos_names.npy', pos_names)
#
neg_rho_wrt_center = np.concatenate(neg_rho_wrt_center, axis=0)
neg_theta_wrt_center = np.concatenate(neg_theta_wrt_center, axis=0)
neg_input_feat = np.concatenate(neg_input_feat, axis=0)
neg_mask = np.concatenate(neg_mask, axis=0)
np.save(params['cache_dir']+'/neg_names.npy', neg_names)
#

print("Read {} negative shapes".format((neg_rho_wrt_center).shape))
print("Read {} positive shapes".format((pos_rho_wrt_center).shape))
np.save(params['cache_dir']+'/target_rho_wrt_center.npy', target_rho_wrt_center)
np.save(params['cache_dir']+'/target_theta_wrt_center.npy', target_theta_wrt_center)
np.save(params['cache_dir']+'/target_input_feat.npy', target_input_feat)
np.save(params['cache_dir']+'/target_mask.npy', target_mask)

np.save(params['cache_dir']+'/pos_training_idx.npy', training_idx)
np.save(params['cache_dir']+'/pos_val_idx.npy', val_idx)
np.save(params['cache_dir']+'/pos_test_idx.npy', test_idx)
np.save(params['cache_dir']+'/pos_rho_wrt_center.npy', pos_rho_wrt_center)
np.save(params['cache_dir']+'/pos_theta_wrt_center.npy', pos_theta_wrt_center)
np.save(params['cache_dir']+'/pos_input_feat.npy', pos_input_feat)
np.save(params['cache_dir']+'/pos_mask.npy', pos_mask)

np.save(params['cache_dir']+'/neg_training_idx.npy', training_idx)
np.save(params['cache_dir']+'/neg_val_idx.npy', val_idx)
np.save(params['cache_dir']+'/neg_test_idx.npy', test_idx)
np.save(params['cache_dir']+'/neg_rho_wrt_center.npy', neg_rho_wrt_center)
np.save(params['cache_dir']+'/neg_theta_wrt_center.npy', neg_theta_wrt_center)
np.save(params['cache_dir']+'/neg_input_feat.npy', neg_input_feat)
np.save(params['cache_dir']+'/neg_mask.npy', neg_mask)
np.save(params['cache_dir']+'/neg_names.npy', neg_names)
