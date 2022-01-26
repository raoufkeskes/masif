import time
import math
from sklearn import metrics
import numpy as np
import sys
import os
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score


# Features and theta are flipped for the binder in construct_batch (except for hydrophobicity).
from tqdm import tqdm


def construct_batch(
        binder_rho_wrt_center,
        binder_theta_wrt_center,
        binder_input_feat,
        binder_mask,
        c_pos_training_idx,
        pos_rho_wrt_center,
        pos_theta_wrt_center,
        pos_input_feat,
        pos_mask,
        c_neg_training_idx,
        neg_rho_wrt_center,
        neg_theta_wrt_center,
        neg_input_feat,
        neg_mask,
        c_neg_training_idx_2=None,
):
    batch_rho_coords_binder = np.expand_dims(
        binder_rho_wrt_center[c_pos_training_idx], 2
    )
    batch_theta_coords_binder = np.expand_dims(
        binder_theta_wrt_center[c_pos_training_idx], 2
    )
    batch_input_feat_binder = binder_input_feat[c_pos_training_idx]
    batch_mask_binder = binder_mask[c_pos_training_idx]

    batch_rho_coords_pos = np.expand_dims(pos_rho_wrt_center[c_pos_training_idx], 2)
    batch_theta_coords_pos = np.expand_dims(pos_theta_wrt_center[c_pos_training_idx], 2)
    batch_input_feat_pos = pos_input_feat[c_pos_training_idx]
    batch_mask_pos = pos_mask[c_pos_training_idx]

    # Negate the input_features of the binder, except the last column.
    batch_input_feat_binder = -batch_input_feat_binder
    # TODO: This should not be like this ... it is a hack.
    if batch_input_feat_binder.shape[2] == 5 or batch_input_feat_binder.shape[2] == 3:
        batch_input_feat_binder[:, :, -1] = -batch_input_feat_binder[
                                             :, :, -1
                                             ]  # Do not negate hydrophobicity.
    # Also negate the theta coords for the binder.
    batch_theta_coords_binder = 2 * np.pi - batch_theta_coords_binder

    batch_rho_coords_neg = np.expand_dims(neg_rho_wrt_center[c_neg_training_idx], 2)
    batch_theta_coords_neg = np.expand_dims(neg_theta_wrt_center[c_neg_training_idx], 2)
    batch_input_feat_neg = neg_input_feat[c_neg_training_idx]
    batch_mask_neg = neg_mask[c_neg_training_idx]

    batch_rho_coords_neg_2 = batch_rho_coords_binder.copy()
    batch_theta_coords_neg_2 = batch_theta_coords_binder.copy()
    batch_input_feat_neg_2 = batch_input_feat_binder.copy()
    batch_mask_neg_2 = batch_mask_binder.copy()

    batch_rho_coords = np.concatenate(
        [
            batch_rho_coords_pos,
            batch_rho_coords_binder,
            batch_rho_coords_neg,
            batch_rho_coords_neg_2,
        ],
        axis=0,
    )
    batch_theta_coords = np.concatenate(
        [
            batch_theta_coords_pos,
            batch_theta_coords_binder,
            batch_theta_coords_neg,
            batch_theta_coords_neg_2,
        ],
        axis=0,
    )
    batch_input_feat = np.concatenate(
        [
            batch_input_feat_pos,
            batch_input_feat_binder,
            batch_input_feat_neg,
            batch_input_feat_neg_2,
        ],
        axis=0,
    )
    batch_mask = np.concatenate(
        [batch_mask_pos, batch_mask_binder, batch_mask_neg, batch_mask_neg_2], axis=0
    )
    # expand the last dimension of the mask (batch_size, max_points_patch, 1)
    batch_mask = np.expand_dims(batch_mask, 2)

    return batch_rho_coords, batch_theta_coords, batch_input_feat, batch_mask


def compute_dists(descs1, descs2):
    dists = np.sqrt(np.sum(np.square(descs1 - descs2), axis=1))
    return dists


def construct_batch_val_test(
        c_idx, rho_wrt_center, theta_wrt_center, input_feat, mask, flip=False
):
    batch_rho_coords = np.expand_dims(rho_wrt_center[c_idx], 2)
    batch_theta_coords = np.expand_dims(theta_wrt_center[c_idx], 2)
    batch_input_feat = input_feat[c_idx]
    batch_mask = mask[c_idx]
    batch_mask = np.expand_dims(batch_mask, 2)
    # Flip features and theta (except hydrophobicity)
    if flip:
        batch_input_feat = -batch_input_feat
        batch_theta_coords = 2 * np.pi - batch_theta_coords
        assert len(batch_input_feat.shape) == 3
        # Hydrophobicity is not flipped. -- FIx this.
        if batch_input_feat.shape[2] == 5 or batch_input_feat.shape[2] == 3:
            batch_input_feat[:, :, -1] = -batch_input_feat[:, :, -1]

    return batch_rho_coords, batch_theta_coords, batch_input_feat, batch_mask


def compute_val_test_desc(
        learning_obj,
        idx,
        rho_wrt_center,
        theta_wrt_center,
        input_feat,
        mask,
        batch_size=100,
        flip=False,
):
    all_descs = []
    num_batches = int(np.ceil(float(len(idx)) / float(batch_size)))
    # Compute all desc for positive shapes.
    for kk in range(num_batches):
        c_idx = idx[np.arange(kk * batch_size, min((kk + 1) * batch_size, len(idx)))]

        batch_rho_coords, batch_theta_coords, batch_input_feat, batch_mask = construct_batch_val_test(
            c_idx, rho_wrt_center, theta_wrt_center, input_feat, mask, flip=flip
        )

        feed_dict = {
            learning_obj.rho_coords: batch_rho_coords,
            learning_obj.theta_coords: batch_theta_coords,
            learning_obj.input_feat: batch_input_feat,
            learning_obj.mask: batch_mask,
            learning_obj.keep_prob: 1.0,
        }

        desc = learning_obj.session.run([learning_obj.global_desc], feed_dict=feed_dict)
        desc = np.squeeze(desc)
        if len(desc.shape) == 1:
            desc = np.expand_dims(desc, 0)
        all_descs.append(desc)

    if len(all_descs) > 1:
        all_descs = np.concatenate(all_descs, axis=0)
    else:
        all_descs = all_descs[0]
    return all_descs


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


# Randomly pick
def train_ppi_search(
        learning_obj,
        params,
        binder_rho_wrt_center,
        binder_theta_wrt_center,
        binder_input_feat,
        binder_mask,
        pos_training_idx,
        pos_val_idx,
        pos_test_idx,
        pos_rho_wrt_center,
        pos_theta_wrt_center,
        pos_input_feat,
        pos_mask,
        neg_training_idx,
        neg_val_idx,
        neg_test_idx,
        neg_rho_wrt_center,
        neg_theta_wrt_center,
        neg_input_feat,
        neg_mask,
        num_iterations=1000000,
        num_iter_test=1000,
        batch_size=32,
        num_epochs=10,
        batch_size_val_test=1000,
):
    print("Number of training positive shapes: {}\n".format(len(pos_training_idx)))
    print("Number of validation positive shapes: {}\n".format(len(pos_val_idx)))
    print("Number of testing positive shapes: {}\n".format(len(pos_test_idx)))

    print("Number of training negative shapes: {}\n".format(len(neg_training_idx)))
    print("Number of validation negative shapes: {}\n".format(len(neg_val_idx)))
    print("Number of testing negative shapes: {}\n".format(len(neg_test_idx)))

    list_training_loss = []
    list_training_norm_grad = []
    iter_time = []
    best_val_auc = 0

    training_idx = np.copy(pos_training_idx)

    iter_training_loss = []

    n_batches = int(np.ceil(len(training_idx) / batch_size))
    for epoch in range(num_epochs):
        print("epoch {} ...".format(epoch))
        np.random.shuffle(training_idx)
        start = 0
        training_loss_ls = []
        pos_dist_ls = []
        neg_dist_ls = []
        for batch in tqdm(range(n_batches)):
            # contruct the batch
            c_pos_training_idx = training_idx[start: start + batch_size]
            c_neg_training_idx = training_idx[start: start + batch_size]
            c_neg_training_idx_2 = None

            # Features and theta are flipped for the binder in construct_batch (except for hydrophobicity).
            batch_rho_coords, batch_theta_coords, batch_input_feat, batch_mask = construct_batch(
                binder_rho_wrt_center,
                binder_theta_wrt_center,
                binder_input_feat,
                binder_mask,
                c_pos_training_idx,
                pos_rho_wrt_center,
                pos_theta_wrt_center,
                pos_input_feat,
                pos_mask,
                c_neg_training_idx,
                neg_rho_wrt_center,
                neg_theta_wrt_center,
                neg_input_feat,
                neg_mask,
                c_neg_training_idx_2,
            )
            feed_dict = {
                learning_obj.rho_coords: batch_rho_coords,
                learning_obj.theta_coords: batch_theta_coords,
                learning_obj.input_feat: batch_input_feat,
                learning_obj.mask: batch_mask,
                learning_obj.keep_prob: 0.5,
            }

            # Do not train during the first iteration
            # scores are the concat of distances
            # scores are of size batch_size/2 ===> [pos_distances, neg_distances]

            if epoch == 0:
                training_loss, score = learning_obj.session.run(
                    [learning_obj.data_loss,
                     learning_obj.score], feed_dict=feed_dict
                )
            else:
                _, training_loss, norm_grad, score = learning_obj.session.run(
                    [
                        learning_obj.optimizer,
                        learning_obj.data_loss,
                        learning_obj.norm_grad,
                        learning_obj.score,
                    ],
                    feed_dict=feed_dict,
                )

            training_loss_ls.append(training_loss)
            n = len(score) // 2
            try:
                pos_score = score[:n]
                neg_score = score[n:]
                print(neg_score)
            except:
                print(score)
                sys.exit(1)
            pos_dist_ls = np.concatenate([pos_score, pos_dist_ls], axis=0)
            neg_dist_ls = np.concatenate([neg_score, neg_dist_ls], axis=0)

            start += batch_size

        print("Training loss : ", np.array(training_loss_ls).mean())
        print("Mean training positive distance: {} ".format(np.mean(pos_dist_ls)))
        print("Mean training negative distance: {} ".format(np.mean(neg_dist_ls)))






        # if num_iter % num_iter_test == 0:
        #     # print("Validating and testing.\n ")
        #     roc_auc = 1 - compute_roc_auc(iter_pos_score, iter_neg_score)
        #     print("training_loss: {}\n".format(np.mean(list_training_loss)))
        #     print("Approx Training ROC AUC: {}\n ".format(roc_auc))
        #     print("Mean training positive score: {} ".format(np.mean(1.0 / iter_pos_score)))
        #     print("Mean training negative score: {} ".format(np.mean(1.0 / iter_neg_score)))
        #     print("Training ROC AUC: {}\n ".format(roc_auc))
        #     #
        #     iter_pos_score = []
        #     iter_neg_score = []
        #     list_training_loss = []
        #
        #     training_time_entry = time.time() - tic
        #     print("Training 1000 entries took {}".format(training_time_entry))
        #     print("pos scores : ", pos_score)
        #     print("neg scores : ", neg_score)
            # tic = time.time()
            # # Compute validation descriptors.
            # pos_desc = compute_val_test_desc(
            #     learning_obj,
            #     pos_val_idx,
            #     pos_rho_wrt_center,
            #     pos_theta_wrt_center,
            #     pos_input_feat,
            #     pos_mask,
            #     batch_size=batch_size_val_test,
            # )
            #
            # binder_desc = compute_val_test_desc(
            #     learning_obj,
            #     pos_val_idx,
            #     binder_rho_wrt_center,
            #     binder_theta_wrt_center,
            #     binder_input_feat,
            #     binder_mask,
            #     batch_size=batch_size_val_test,
            #     flip=True,
            # )
            #
            # neg_desc = compute_val_test_desc(
            #     learning_obj,
            #     neg_val_idx,
            #     neg_rho_wrt_center,
            #     neg_theta_wrt_center,
            #     neg_input_feat,
            #     neg_mask,
            #     batch_size=batch_size_val_test,
            # )
            #
            # neg_desc_2 = binder_desc.copy()
            # # Simply shuffle negative descriptors.
            # np.random.shuffle(neg_desc)
            #
            # # Compute val ROC AUC.
            # pos_dists = compute_dists(pos_desc, binder_desc)
            # neg_dists = compute_dists(neg_desc, neg_desc_2)
            # try:
            #     val_auc = 1 - compute_roc_auc(pos_dists, neg_dists)
            # except:
            #     print(np.min(pos_dists))
            #     print(np.min(neg_dists))
            #     sys.exit(1)
            # val_time = time.time() - tic
            #
            # print("Iteration {} validation roc auc: {}\n".format(num_iter, val_auc))
            # print("Iteration {} validation roc auc: {}".format(num_iter, val_auc))
            # print("Mean validation positive score: {} ".format(np.mean(pos_dists)))
            # print("Mean validation negative score: {} ".format(np.mean(neg_dists)))
            # # Compute TEST ROC AUC.

            # tic = time.time()
            # pos_desc = compute_val_test_desc(
            #     learning_obj,
            #     pos_test_idx,
            #     pos_rho_wrt_center,
            #     pos_theta_wrt_center,
            #     pos_input_feat,
            #     pos_mask,
            #     batch_size=batch_size_val_test,
            # )
            #
            # binder_desc = compute_val_test_desc(
            #     learning_obj,
            #     pos_test_idx,
            #     binder_rho_wrt_center,
            #     binder_theta_wrt_center,
            #     binder_input_feat,
            #     binder_mask,
            #     batch_size=batch_size_val_test,
            #     flip=True,
            # )
            #
            # neg_desc = compute_val_test_desc(
            #     learning_obj,
            #     neg_test_idx,
            #     neg_rho_wrt_center,
            #     neg_theta_wrt_center,
            #     neg_input_feat,
            #     neg_mask,
            #     batch_size=batch_size_val_test,
            # )
            #
            # neg_desc_2 = binder_desc.copy()
            #
            # # Compute test ROC AUC.
            # pos_dists = compute_dists(pos_desc, binder_desc)
            # np.random.shuffle(neg_desc)
            # neg_dists = compute_dists(neg_desc, neg_desc_2)
            # test_auc = 1 - compute_roc_auc(pos_dists, neg_dists)
            #
            # test_time = time.time() - tic
            # print("Iteration {} test roc auc: {}\n".format(num_iter, test_auc))
            # print("Iteration time: {} validation time: {} test time: {}\n".format(np.mean(np.array(iter_time)),
            #                                                                       val_time,
            #                                                                       test_time))
            # print("Iteration {} test roc auc: {}".format(num_iter, test_auc))
            # print("Iteration time: {} validation time: {} test time: {}".format(np.mean(np.array(iter_time)),
            #                                                                     val_time,
            #                                                                     test_time))
            # print("Mean test positive score: {} ".format(np.mean(pos_dists)))
            # print("Mean test negative score: {} ".format(np.mean(neg_dists)))
            #
            # tic = time.time()
            #
            # out_dir = params["model_dir"]
            # if val_auc > best_val_auc:
            #     print(">>> Saving model.")
            #     best_val_auc = val_auc
            #     output_model = out_dir + "model"
            #     learning_obj.saver.save(learning_obj.session, output_model)
            #     np.save(out_dir + "pos_dists.npy", pos_dists)
            #     np.save(out_dir + "pos_test_idx.npy", neg_test_idx)
            #     np.save(out_dir + "neg_dists.npy", neg_dists)
            #     np.save(out_dir + "pos_desc.npy", pos_desc)
            #     np.save(out_dir + "binder_desc.npy", binder_desc)
            #     np.save(out_dir + "neg_desc.npy", neg_desc)
            #     np.save(out_dir + "neg_test_idx.npy", neg_test_idx)
            #     np.save(out_dir + "neg_desc_2.npy", neg_desc_2)
            #     np.save(out_dir + "neg_desc_2.npy", neg_desc_2)

            # print("----------------------------------------------------")
