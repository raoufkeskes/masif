import time
import os
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import pickle
from tqdm.auto import tqdm



# Apply mask to input_feat
from masif_modules.protein import Protein


def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def pad_indices(indices, max_verts):
    padded_ix = np.zeros((len(indices), max_verts), dtype=int)
    for patch_ix in range(len(indices)):
        padded_ix[patch_ix] = np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))]
        )
    return padded_ix


# Run masif site on a protein, on a previously trained network.
def run_masif_site(
        params, learning_obj, rho_wrt_center, theta_wrt_center, input_feat, mask, indices
):
    indices = pad_indices(indices, mask.shape[1])
    mask = np.expand_dims(mask, 2)
    feed_dict = {
        learning_obj.rho_coords: rho_wrt_center,
        learning_obj.theta_coords: theta_wrt_center,
        learning_obj.input_feat: input_feat,
        learning_obj.mask: mask,
        learning_obj.indices_tensor: indices,
    }

    score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
    return score


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


def train_masif_site(
        learning_obj,
        params,
        batch_size=100,
        num_iterations=100,
        num_iter_test=1000,
        batch_size_val_test=50,
):
    # Open training list.

    list_training_loss = []
    list_training_auc = []
    list_validation_auc = []
    iter_time = []
    best_val_auc = 0

    out_dir = params["model_dir"]
    logfile = open(out_dir + "log.txt", "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    df = pd.read_csv("/home/raouf_ks/Desktop/Projects/structures_data/tables/mabsite_only_proteins_antibodies.csv")

    print(df.shape)

    df_train = df[df["set"] == "train"]
    df_valid = df[df["set"] == "valid"]

    df_train_valid = pd.concat((df_train, df_valid))
    df_test = df[df["set"] == "test"]

    precomp_path = "/home/raouf_ks/Desktop/data/precomputed/"
    radius = 9

    for num_iter in range(num_iterations):
        # Start training epoch:
        list_training_loss = []
        list_training_auc = []
        list_val_auc = []
        list_val_pos_labels = []
        list_val_neg_labels = []
        list_val_names = []
        list_training_acc = []
        list_val_acc = []
        print("Starting epoch {} \n".format(num_iter))
        tic = time.time()
        all_training_labels = []
        all_training_scores = []
        all_val_labels = []
        all_val_scores = []
        all_test_labels = []
        all_test_scores = []
        count_proteins = 0

        list_test_auc = []
        list_test_names = []
        list_test_acc = []
        all_test_labels = []
        all_test_scores = []

        pbar1 = tqdm(iterable=df_train_valid.iterrows(), total=len(df_train_valid), desc="training ... ")
        for idx, row in pbar1:
            pbar1.set_description("training ... " + row["antibody_filename"])
            pkl_path = os.path.join(precomp_path, str(radius) + "A",
                                   row["antibody_filename"] + ".pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    protein = Protein(*pickle.load(f))
                    protein.normalize_hydrophobia()
                    protein.normalize_electrostatics()
                    # protein.patches_features = protein.patches_features[:, :, [0, 1, 4, 3, 2]]
                    iface_labels = protein.iface
                    if (
                            np.sum(iface_labels) > 0.75 * len(iface_labels)
                            or np.sum(iface_labels) < 30
                    ):
                        continue
                    count_proteins += 1

                    rho_wrt_center = protein.rhos
                    theta_wrt_center = protein.thetas
                    input_feat = protein.patches_features
                    if np.sum(params["feat_mask"]) < 5:
                        input_feat = mask_input_feat(input_feat, params["feat_mask"])
                    mask = protein.mask
                    mask = np.expand_dims(mask, 2)
                    indices = protein.neigh_indices
                    # indices is (n_verts x <30), it should be
                    indices = pad_indices(indices, mask.shape[1])
                    tmp = np.zeros((len(iface_labels), 2))
                    for i in range(len(iface_labels)):
                        if iface_labels[i] == 1:
                            tmp[i, 0] = 1
                        else:
                            tmp[i, 1] = 1
                    iface_labels_dc = tmp
                    pos_labels = np.where(iface_labels == 1)[0]
                    neg_labels = np.where(iface_labels == 0)[0]
                    np.random.shuffle(neg_labels)
                    np.random.shuffle(pos_labels)
                    # Scramble neg idx, and only get as many as pos_labels to balance the training.
                    if params["n_conv_layers"] == 1:
                        n = min(len(pos_labels), len(neg_labels))
                        n = min(n, batch_size // 2)
                        subset = np.concatenate([neg_labels[:n], pos_labels[:n]])

                        rho_wrt_center = rho_wrt_center[subset]
                        theta_wrt_center = theta_wrt_center[subset]
                        input_feat = input_feat[subset]
                        mask = mask[subset]
                        iface_labels_dc = iface_labels_dc[subset]
                        indices = indices[subset]
                        pos_labels = range(0, n)
                        neg_labels = range(n, n * 2)
                    else:
                        n = min(len(pos_labels), len(neg_labels))
                        neg_labels = neg_labels[:n]
                        pos_labels = pos_labels[:n]

                    feed_dict = {
                        learning_obj.rho_coords: rho_wrt_center,
                        learning_obj.theta_coords: theta_wrt_center,
                        learning_obj.input_feat: input_feat,
                        learning_obj.mask: mask,
                        learning_obj.labels: iface_labels_dc,
                        learning_obj.pos_idx: pos_labels,
                        learning_obj.neg_idx: neg_labels,
                        learning_obj.indices_tensor: indices,
                    }

                    if row["set"] == "train":
                        feed_dict[learning_obj.keep_prob] = 1.0
                        _, training_loss, norm_grad, score, eval_labels = learning_obj.session.run(
                            [
                                learning_obj.optimizer,
                                learning_obj.data_loss,
                                learning_obj.norm_grad,
                                learning_obj.eval_score,
                                learning_obj.eval_labels,
                            ],
                            feed_dict=feed_dict,
                        )
                        all_training_labels = np.concatenate(
                            [all_training_labels, eval_labels[:, 0]]
                        )
                        all_training_scores = np.concatenate([all_training_scores, score])
                        auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                        list_training_auc.append(auc)
                        list_training_loss.append(np.mean(training_loss))

                    else:
                        # pass
                        # logfile.write("Validating on {} {}\n".format(row["antigen_filename"]))
                        feed_dict[learning_obj.keep_prob] = 1.0
                        training_loss, score, eval_labels = learning_obj.session.run(
                            [
                                learning_obj.data_loss,
                                learning_obj.eval_score,
                                learning_obj.eval_labels,
                            ],
                            feed_dict=feed_dict,
                        )
                        auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                        list_val_pos_labels.append(np.sum(iface_labels))
                        list_val_neg_labels.append(len(iface_labels) - np.sum(iface_labels))
                        list_val_auc.append(auc)
                        all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0]])
                        all_val_scores = np.concatenate([all_val_scores, score])
        # Run testing cycle.
        # for idx, row in df_test.iterrows():
        #     with open(os.path.join(precomp_path, str(radius) + "A",
        #                            row["antigen_filename"] + ".pkl"), "rb") as f:
        #         protein = pickle.load(f)
        #         protein.normalize_hydrophobia()
        #         protein.normalize_electrostatics()
        #         # protein.patches_features = protein.patches_features[:, :, [0, 1, 4, 3, 2]]
        #         logfile.write("Testing on {}\n".format(row["antigen_filename"]))
        #         iface_labels = protein.iface
        #         if (len(iface_labels) == 0) or (len(iface_labels) > 20000):
        #             continue
        #         if (
        #                 np.sum(iface_labels) > 0.75 * len(iface_labels)
        #                 or np.sum(iface_labels) < 30
        #         ):
        #             continue
        #         count_proteins += 1
        #
        #         rho_wrt_center = protein.rhos
        #         theta_wrt_center = protein.thetas
        #         input_feat = protein.patches_features
        #         if np.sum(params["feat_mask"]) < 5:
        #             input_feat = mask_input_feat(input_feat, params["feat_mask"])
        #         mask = protein.mask
        #         mask = np.expand_dims(mask, 2)
        #         indices = protein.neigh_indices
        #         # indices is (n_verts x <30), it should be
        #         indices = pad_indices(indices, mask.shape[1])
        #         tmp = np.zeros((len(iface_labels), 2))
        #         for i in range(len(iface_labels)):
        #             if iface_labels[i] == 1:
        #                 tmp[i, 0] = 1
        #             else:
        #                 tmp[i, 1] = 1
        #         iface_labels_dc = tmp
        #         logfile.flush()
        #         pos_labels = np.where(iface_labels == 1)[0]
        #         neg_labels = np.where(iface_labels == 0)[0]
        #
        #         feed_dict = {learning_obj.rho_coords: rho_wrt_center, learning_obj.theta_coords: theta_wrt_center,
        #                      learning_obj.input_feat: input_feat, learning_obj.mask: mask,
        #                      learning_obj.labels: iface_labels_dc, learning_obj.pos_idx: pos_labels,
        #                      learning_obj.neg_idx: neg_labels, learning_obj.indices_tensor: indices,
        #                      learning_obj.keep_prob: 1.0}
        #
        #         score = learning_obj.session.run(
        #             [learning_obj.full_score], feed_dict=feed_dict
        #         )
        #         score = score[0]
        #         auc = metrics.roc_auc_score(iface_labels, score)
        #         list_test_auc.append(auc)
        #         list_test_names.append(row["antigen_filename"])
        #         all_test_labels.append(iface_labels)
        #         all_test_scores.append(score)

        print("training loss : {:.6f}  Per protein AUC mean (training): {:.4f}({:.4f}); median: {:.4f}({:.4f}) \n".format(np.mean(list_training_loss),
                                                                                                                 np.mean(list_training_auc),
                                                                                                                np.mean(list_val_auc),
                                                                                                                np.median(list_training_auc),
                                                                                                                 np.median(list_val_auc)))

        output_model = "/home/raouf_ks/Desktop/Projects/MAbSIF/pretrained_models/tf_models/antibody/"+str(num_iter)+"/model"
        learning_obj.saver.save(learning_obj.session, output_model, global_step=num_iter)

        # if np.mean(list_val_auc) > best_val_auc:
        #     logfile.write(">>> Saving model.\n")
        #     print(">>> Saving model.\n")
        #     best_val_auc = np.mean(list_val_auc)
        #     output_model = out_dir + "model"
        #     learning_obj.saver.save(learning_obj.session, output_model)
        #     # Save the scores for test.
        #     np.save(out_dir + "test_labels.npy", all_test_labels)
        #     np.save(out_dir + "test_scores.npy", all_test_scores)
        #     np.save(out_dir + "test_names.npy", list_test_names)
