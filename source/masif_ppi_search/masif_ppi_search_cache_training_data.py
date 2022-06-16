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

df = pd.read_csv("/home/raouf-ks/Desktop/Projects/structures_data/tables/mabsite_only_proteins_antibodies.csv")

df_train = df[df["set"] == "train"]
df_valid = df[df["set"] == "valid"]

df_train_valid = pd.concat((df_train, df_valid))
df_test = df[df["set"] == "test"]

batch_size = 32
precomp_path = "/media/raouf-ks/Maxtor/raouf/data/precomputed/"
radius = 12 #9
mabsif_pred_pth = os.path.join("/media/raouf-ks/Maxtor/raouf/data/mabsite/", "9A", "prediction")
mabsif_gt_pth = os.path.join("/media/raouf-ks/Maxtor/raouf/data/mabsite/", "9A", "ground_truth")
mabsite_th = 0.6
mabsif_results_pth = os.path.join("/media/raouf-ks/Maxtor/raouf/data/mabsite/", str(radius) + "A", "prediction")
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
            iface_labels = antibody.iface["kdtree"]
            if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 10
            ):
                continue

            # pos_labels: points > max_sc_filt and >  min_sc_filt.
            pos_labels = np.where(iface_labels == 1)[0]

            K = int(len(pos_labels))
            l = np.arange(len(pos_labels))
            np.random.shuffle(l)
            l = l[:K]
            l = pos_labels[l]

            v1 = antibody.vertices[l]
            v2 = antigen.vertices

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

            mabsite_preds = np.load(os.path.join(mabsif_pred_pth, row["antigen_filename"] + ".npy")).reshape(-1)
            mabsite_gt = antigen.iface["kdtree"]
            # print(len(mabsite_gt))
            # print(len(mabsite_preds))
            # print("--------")
            args1 = (np.argwhere(mabsite_preds <= 0.4)[:, 0])
            args2 = (np.argwhere(mabsite_gt <= 0.4)[:, 0])

            very_neg = np.array(list(set(args1).intersection(set(args2))))

            selected_preds = np.argwhere(mabsite_preds >= 0.7)[:, 0]
            # 3) if no intersection keep them  else make half half
            pseudo_pos = np.array(list(set(k2_neg).intersection(set(selected_preds))))

            # print("anchors : ", len(k1))
            # print("true pos : ", len(k2_pos))
            # print("very neg  : ", len(very_neg))
            # print("pseudo pos  : ", len(pseudo_pos))

            np.random.shuffle(very_neg)
            np.random.shuffle(pseudo_pos)

            sum_lens = len(very_neg) + len(pseudo_pos)
            # print("len very neg ", len(very_neg))
            # print("len pseudo pos ", len(pseudo_pos))
            if sum_lens < len(k1):
                k1, k2_pos = k1[:sum_lens], k2_pos[:sum_lens]
                k2_neg = np.concatenate((very_neg, pseudo_pos))
            else:
                if len(very_neg) <= int(len(k1) / 2):
                    k2_neg = np.concatenate((very_neg[:len(very_neg)], pseudo_pos[:len(k1) - len(very_neg)]))
                elif len(pseudo_pos) <= int(len(k1) / 2):
                    k2_neg = np.concatenate((very_neg[:len(k1) - len(pseudo_pos)], pseudo_pos[:len(pseudo_pos)]))
                else:
                    lower_b, upper_b = int(np.floor((len(k1)) / 2)), int(np.ceil((len(k1)) / 2))
                    k2_neg = np.concatenate((very_neg[:lower_b], pseudo_pos[:upper_b]))

            k2_neg = k2_neg.astype(int)

            k1 = k1[:batch_size]
            k2_pos = k2_pos[:batch_size]
            k2_neg = k2_neg[:batch_size]
            ############################
            # print("len k1 :", len(k1))
            # print("len k2_pos :", len(k2_pos))
            # print("len k2_neg :", len(k2_neg))
            assert (len(k1) == len(k2_pos))
            assert (len(k1) == len(k2_neg))
            n_pos = len(k1)

            # antibody.save_ply_masif(patches_idxs=[k1[0]])
            # antigen.save_ply_masif(patches_idxs=[k2_pos[0]])
            # antigen.save_ply_masif(patches_idxs=[k2_neg[0]])
            # antigen.save_ply_masif(iface=mabsite_preds, name_extension="_pred")

            #
            ppi_pair_id = row["complex_filename"]
            pid = row["antigen_filename"]  # target is p1
            for ii in k1:
                pos_names.append('{}_{}_{}'.format(ppi_pair_id, pid, ii))

            target_rho_wrt_center.append(antibody.rhos[k1])
            target_theta_wrt_center.append(antibody.thetas[k1])
            target_input_feat.append(antibody.patches_features[k1])
            target_mask.append(antibody.mask[k1])

            # Read as positives those points.
            pos_rho_wrt_center.append(antigen.rhos[k2_pos])
            pos_theta_wrt_center.append(antigen.thetas[k2_pos])
            pos_input_feat.append(antigen.patches_features[k2_pos])
            pos_mask.append(antigen.mask[k2_pos])

            # TO DO
            # shuffle pseudo pos shuffle neg and sample len(k2_pos)/2 for both
            np.random.shuffle(k2_neg)
            k2_neg = k2_neg[:(len(k2_pos))]
            assert (len(k2_neg) == n_pos)

            neg_rho_wrt_center.append(antigen.rhos[k2_neg])
            neg_theta_wrt_center.append(antigen.thetas[k2_neg])
            neg_input_feat.append(antigen.patches_features[k2_neg])
            neg_mask.append(antigen.mask[k2_neg])

            pid = row["antibody_filename"]
            for ii in k2_neg:
                neg_names.append('{}_{}_{}'.format(ppi_pair_id, pid, ii))

            # Training, validation or test?

            if row["set"] == "train":
                training_idx = np.append(training_idx, np.arange(idx_count, idx_count + n_pos))
            elif row["set"] == "valid":
                val_idx = np.append(val_idx, np.arange(idx_count, idx_count + n_pos))
            else:
                test_idx = np.append(test_idx, np.arange(idx_count, idx_count + n_pos))

            idx_count += n_pos

params['cache_dir'] = "/media/raouf-ks/Maxtor/raouf/masif_cache"

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
np.save(params['cache_dir'] + '/pos_names.npy', pos_names)

neg_rho_wrt_center = np.concatenate(neg_rho_wrt_center, axis=0)
neg_theta_wrt_center = np.concatenate(neg_theta_wrt_center, axis=0)
neg_input_feat = np.concatenate(neg_input_feat, axis=0)
neg_mask = np.concatenate(neg_mask, axis=0)
np.save(params['cache_dir'] + '/neg_names.npy', neg_names)

print("Read {} negative shapes".format(neg_rho_wrt_center.shape))
print("Read {} positive shapes".format(pos_rho_wrt_center.shape))
np.save(params['cache_dir'] + '/target_rho_wrt_center.npy', target_rho_wrt_center)
np.save(params['cache_dir'] + '/target_theta_wrt_center.npy', target_theta_wrt_center)
np.save(params['cache_dir'] + '/target_input_feat.npy', target_input_feat)
np.save(params['cache_dir'] + '/target_mask.npy', target_mask)

np.save(params['cache_dir'] + '/pos_training_idx.npy', training_idx)
np.save(params['cache_dir'] + '/pos_val_idx.npy', val_idx)
np.save(params['cache_dir'] + '/pos_test_idx.npy', test_idx)
np.save(params['cache_dir'] + '/pos_rho_wrt_center.npy', pos_rho_wrt_center)
np.save(params['cache_dir'] + '/pos_theta_wrt_center.npy', pos_theta_wrt_center)
np.save(params['cache_dir'] + '/pos_input_feat.npy', pos_input_feat)
np.save(params['cache_dir'] + '/pos_mask.npy', pos_mask)

np.save(params['cache_dir'] + '/neg_training_idx.npy', training_idx)
np.save(params['cache_dir'] + '/neg_val_idx.npy', val_idx)
np.save(params['cache_dir'] + '/neg_test_idx.npy', test_idx)
np.save(params['cache_dir'] + '/neg_rho_wrt_center.npy', neg_rho_wrt_center)
np.save(params['cache_dir'] + '/neg_theta_wrt_center.npy', neg_theta_wrt_center)
np.save(params['cache_dir'] + '/neg_input_feat.npy', neg_input_feat)
np.save(params['cache_dir'] + '/neg_mask.npy', neg_mask)
np.save(params['cache_dir'] + '/neg_names.npy', neg_names)
