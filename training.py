import torch
import copy
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import TaskOrganizedDataset
from utils import (compute_matrices, avg_accuracy, avg_forgetting, forward_transfer, backward_transfer, print_metrics,
                   HammingDistance, MaskedTripletMarginLoss, pearson_corr, matthews_corr, raw_counts)

import pytorch_metric_learning.losses, pytorch_metric_learning.miners
from metrics import concept_alignment_score

import numpy as np

def train(net: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module],
          train_set: TaskOrganizedDataset,
          val_set: TaskOrganizedDataset,
          test_set: TaskOrganizedDataset,
          opts: dict) -> tuple[dict[str, list | str], dict[str, list | str], dict[str, list | str]]:
    """Train a neural-network-based model on a dataset composed of multiple tasks.

        :param net: The neural network to use (where the number of output neurons is equal to the numer of tasks) or
            a list/tuple of neural networks (in case of independent, per-task models).
        :param train_set: The multi-task training set.
        :param val_set: The multi-task validation set.
        :param test_set: The multi-task test set.
        :param opts: The option dictionary.
        :returns: Three dictionaries with metrics (about train set, val set, and test set, respectively).
    """

    # checking
    num_tasks = train_set.num_tasks
    assert num_tasks == val_set.num_tasks and num_tasks == test_set.num_tasks, \
        'Unmatched number of tasks among train, val, test sets (this code assumes the number of tasks is the same).'
    independent_nets = None

    # preparing the list of training sets and the list of networks, in function of the selected training mode
    if opts['train'] == 'joint':
        train_sets = [train_set]
        net.to(torch.device(opts['device']))
        assert opts['replay_buffer'] == 0 and opts['replay_lambda'] == 0., \
            "Options 'replay_buffer' and 'replay_lambda' are expected to be set to zero when training " \
            "in a 'joint' manner."
    elif opts['train'] == 'independent':
        train_sets = train_set.get_task_datasets()
        assert isinstance(net, (tuple, list)), \
            "When training independent models, the 'net' arguments is expected to be a list/tuple of models."
        assert len(net) == num_tasks, \
            "The number of net models must be the same as the number of tasks."
        assert opts['replay_buffer'] == 0 and opts['replay_lambda'] == 0., \
            "Options 'replay_buffer' and 'replay_lambda' are expected to be set to zero when training " \
            "in an 'independent' manner."
        independent_nets = net
    elif opts['train'] == 'continual_task':
        train_sets = train_set.get_task_datasets()
        net.to(torch.device(opts['device']))
    elif opts['train'] == 'continual_online':
        train_sets = train_set.get_task_datasets()
        net.to(torch.device(opts['device']))
    else:
        raise ValueError("Unknown value for 'train' option: " + str(opts['train']))

    # checking task distribution across training set (right now, only disjoint sets of tasks are supported)
    ensure_disjoint_tasks_among_train_sets = {}
    for t in train_sets:
        for task_id in t.task_ids:
            assert task_id not in ensure_disjoint_tasks_among_train_sets, \
                'Overlapping task IDs among training sets, unsupported.'
            ensure_disjoint_tasks_among_train_sets[task_id] = True

    # checking again
    assert len(train_sets) == num_tasks or len(train_sets) == 1, \
        "Assuming the number of training sets to be equal to the number of tasks, or to have a single training set."

    # metrics
    metrics_train = {
        'name': 'train',
        'acc_matrix': -torch.ones(num_tasks, num_tasks, dtype=torch.float),
        'avg_accuracy': [-1.] * num_tasks,
        'avg_forgetting': [-1.] * num_tasks,
        'backward_transfer': [-1.] * num_tasks,
        'forward_transfer': [-1.] * num_tasks,
        'cas': [-1.] * num_tasks,
        'tas': [-1.] * num_tasks,
        'cas_extended': [-1.] * num_tasks,
        'tas_extended': [-1.] * num_tasks,

    }

    if opts['correlate_each_task']:
        #metrics_train['concept_correlation_pearson_pp_continual'] = []
        #metrics_train['concept_correlation_pearson_pt_continual'] = []
        metrics_train['concept_correlation_phi_pp_continual'] = []
        metrics_train['concept_correlation_phi_pt_continual'] = []
        #metrics_train['counts_pt_continual'] = []
        #metrics_train['concept_correlation_pearson_pp_continual_extended'] = []
        #metrics_train['concept_correlation_pearson_pt_continual_extended'] = []
        metrics_train['concept_correlation_phi_pp_continual_extended'] = []
        metrics_train['concept_correlation_phi_pt_continual_extended'] = []
        #metrics_train['counts_pt_continual_extended'] = []

    metrics_val = copy.deepcopy(metrics_train)
    metrics_val['name'] = 'val'
    metrics_test = copy.deepcopy(metrics_train)
    metrics_test['name'] = 'test'

    # Training set only metrics.
    metrics_train['loss'] = []
    metrics_train['cls_loss'] = []
    metrics_train['concept_loss'] = []
    metrics_train['concept_pol_loss'] = []
    metrics_train['mask_pol_loss'] = []
    metrics_train['triplet_loss_batch'] = []
    metrics_train['triplet_loss_buffer'] = []
    metrics_train['replay_loss'] = []

    extended_concept_vectors = {}

    optimizer = None
    replay_set_data_loader = None
    replay_set_iter = None
    zero_five = torch.ones((opts['batch'], opts['n_concepts'])).to(opts['device']) * .5

    distance_fn = HammingDistance(emb_type='01', use_mask=opts['use_mask'])
    hamming_loss_fn = MaskedTripletMarginLoss(
                                                        margin=opts['hamming_margin'],
                                                        distance=distance_fn,
                                                        reducer=pytorch_metric_learning.reducers.AvgNonZeroReducer())

    mining_fn = pytorch_metric_learning.miners.TripletMarginMiner(margin=opts['hamming_margin'],
                                                                  distance=HammingDistance('01', use_mask='no'),
                                                                  type_of_triplets="semihard")
    # For simplicity, the miner does not use any mask to extract triples.

    img, _, _, _, _, _, _, _ = train_set[0]
    img_shape = img.shape

    x_buff_batch = torch.zeros((opts['batch'], *img_shape), dtype=torch.float).to(opts['device'])


    # loop on the provided training sets
    for train_task_id in range(0, len(train_sets)):

        # selecting the training set (possibly balancing positive and negative examples per task)
        if not opts['balance'] or opts['train'] == 'continual_online':
            num_training_examples = len(train_sets[train_task_id])

            train_set_data_loader = \
                DataLoader(train_sets[train_task_id],
                           batch_size=opts['batch'],
                           shuffle=opts['train'] != 'continual_online')
        else:
            balanced_indices = train_sets[train_task_id].get_balanced_sample_indices()
            num_training_examples = len(balanced_indices)

            train_set_data_loader = \
                DataLoader(train_sets[train_task_id],
                           batch_size=opts['batch'],
                           sampler=SubsetRandomSampler(balanced_indices))

        # preparing the (selected) network, if needed
        if opts['train'] == "independent":
            net = independent_nets[train_task_id]
            net.to(torch.device(opts['device']))

        # setting-up train mode
        net.train()

        # optimizer (more optimizers in the case of 'independent' models)
        if (opts['train'] != "independent" and train_task_id == 0) or opts['train'] == "independent":
            if opts['lr'] < 0.:
                optimizer = torch.optim.Adam((p for p in net.parameters() if p.requires_grad),
                                             lr=-opts['lr'], weight_decay=opts['weight_decay'])
            else:
                optimizer = torch.optim.SGD((p for p in net.parameters() if p.requires_grad),
                                            lr=opts['lr'], weight_decay=opts['weight_decay'])

        # epochs on the currently selected training set (it is always equal to one in the 'continual_online' case)
        num_task_epochs = opts['task_epochs'] if opts['train'] != 'continual_online' else 1
        for task_epoch in range(0, num_task_epochs):

            # loop on training samples
            n = 0
            avg_loss = 0.
            loss = 0.
            cls_loss = 0.
            concept_loss = 0.
            concept_pol_loss = 0.
            mask_pol_loss = 0.
            triplet_loss_batch = 0.
            triplet_loss_buffer = 0.
            replay_loss = 0.

            for (x, y, _, true_concepts, stored_concepts, eq_classes, zero_based_train_task_id, abs_idx) in train_set_data_loader:

                # moving data and casting
                x = x.to(opts['device'])
                y = y.to(torch.float32).to(opts['device'])
                # c = concepts.to(torch.float32).to(opts['device'])

                # prediction
                c_pred, c_embs, o = net(x)
                if opts['train'] != 'joint':
                    o = o[:, train_task_id] if o.shape[1] > 1 else o[:, 0]
                else:
                    zero_based_train_task_id = zero_based_train_task_id.to(opts['device'])
                    o = o.gather(1, zero_based_train_task_id.unsqueeze(-1)).squeeze(-1)

                # loss evaluation (from raw outputs)
                cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(o, y, reduction='mean') # Task loss.
                loss = opts['cls_lambda'] * torch.nan_to_num(cls_loss)

                cls_loss = cls_loss.item()

                positive_samples = torch.nonzero(y == 1).reshape((-1,))

                if opts['concept_lambda'] > 0. and opts['min_pos_concepts'] > 0:
                    concept_loss = torch.mean(
                                torch.clamp(float(opts['min_pos_concepts']) - torch.sum(c_pred[positive_samples]), 0))
                    loss += opts['concept_lambda'] * torch.nan_to_num(concept_loss)

                    concept_loss = concept_loss.item()


                if opts['concept_polarization_lambda'] > 0.:
                    concept_pol_loss = (1. - 2. * torch.nn.functional.l1_loss(c_pred,
                                                        zero_five[:c_pred.shape[0],:],
                                                        reduction="mean"))
                    loss += opts['concept_polarization_lambda'] * torch.nan_to_num(concept_pol_loss)

                    concept_pol_loss = concept_pol_loss.item()



                if opts['mask_polarization_lambda'] > 0. and opts['use_mask'] == 'fuzzy' and len(positive_samples) > 0:
                    mask, _ = distance_fn.soft_intersection(c_pred[positive_samples])
                    mask_pol_loss = (1. - 2. * torch.nn.functional.l1_loss(mask, zero_five[:mask.shape[0],:],
                                                        reduction="mean"))
                    loss += opts['mask_polarization_lambda'] * torch.nan_to_num(mask_pol_loss)

                    mask_pol_loss = mask_pol_loss.item()


                # Hamming loss:
                if opts['triplet_lambda'] > 0.:

                    triplet_loss = 0.

                    if opts['batch'] > 3:
                        indices_tuple = mining_fn(c_pred, eq_classes)
                        triplet_loss_batch = hamming_loss_fn(c_pred, eq_classes, indices_tuple=indices_tuple,
                                                        positives=c_pred[positive_samples])

                        triplet_loss = torch.nan_to_num(triplet_loss_batch)

                        triplet_loss_batch = triplet_loss_batch.item()

                    if opts['replay_buffer'] > 2 and task_epoch > 0: # Do not compute triplet loss, unless at least one epoch has passed.
                        triplet_p = torch.zeros((opts['batch'], opts['n_concepts']), dtype=torch.float).to(opts['device']).detach()
                        triplet_n = torch.zeros(triplet_p.shape, dtype=torch.float).to(opts['device']).detach()
                        if len(train_set.buffered_indices) > 0 and \
                            train_task_id in train_set.task2buffered_positives and \
                            train_task_id in train_set.task2buffered_negatives and \
                            positive_samples.shape[0] > 0 and \
                            len(train_set.task2buffered_positives[train_task_id]) > 0 and \
                            len(train_set.task2buffered_negatives[train_task_id]) > 0:
                            if train_task_id == train_task_id: # If we are on the current task_id, fetch anchors from the batch...
                                triplet_a = c_pred[positive_samples]
                                for i in range(c_pred[positive_samples].shape[0]):
                                    _, _, _, _, triplet_p[i,:], _, _, _ = train_set.get_random_buffered_sample(train_task_id, True)
                                    _, _, _, _, triplet_n[i,:], _, _, _ = train_set.get_random_buffered_sample(train_task_id, False)

                                triplet_p = triplet_p.to(opts['device']).detach()
                                triplet_n = triplet_n.to(opts['device']).detach()
                            else: # ...Otherwise, fetch anchors from replay buffer.
                                if opts['model'] == 'mlp':
                                    x_buff_batch[:, :] = 0.
                                else:
                                    x_buff_batch[:, :, :, :] = 0.


                                for i in range(opts['batch']):
                                    if opts['model'] == 'mlp':
                                        x_buff_batch[i, :], triplet_p[i, :], triplet_n[i,
                                                                          :] = train_set.get_random_buffered_triple(
                                            train_task_id)
                                    else:
                                        x_buff_batch[i, :, :, :], triplet_p[i, :], triplet_n[i,
                                                                          :] = train_set.get_random_buffered_triple(
                                            train_task_id)

                                triplet_a, _, _ = net(x_buff_batch)


                            if opts['use_mask'] == 'crisp':
                                _, mask = distance_fn.soft_intersection(triplet_p)
                            elif opts['use_mask'] == 'fuzzy':
                                mask, _ = distance_fn.soft_intersection(triplet_p)
                            else:
                                mask = None

                            ap = distance_fn.hamming_distance_01_masked(triplet_a, triplet_p[:triplet_a.shape[0],:], mask)
                            an = distance_fn.hamming_distance_01_masked(triplet_a, triplet_n[:triplet_a.shape[0],:], mask)


                            current_margins = distance_fn.margin(ap, an)
                            violation = current_margins + hamming_loss_fn.margin


                            if hamming_loss_fn.smooth_loss:
                                loss_mat = torch.nn.functional.softplus(violation)
                            else:
                                loss_mat = torch.nn.functional.relu(violation)

                            triplet_loss_buffer = torch.mean(loss_mat[torch.gt(loss_mat, 0.)]) # AvgNonZero reduction.

                            triplet_loss = torch.nan_to_num(triplet_loss_buffer) + triplet_loss

                            if not isinstance(triplet_loss_buffer, float):
                                triplet_loss_buffer = triplet_loss_buffer.item()

                    if opts['triplet_lambda'] > 0. and opts['replay_buffer'] > 2:
                        triplet_loss /= 2.

                    loss += opts['triplet_lambda'] * triplet_loss

                # experience replay
                if opts['replay_buffer'] > 0:
                    if opts['replay_lambda'] > 0.:

                        # if something is already stored into the memory buffer...
                        if len(train_set.buffered_indices) > 0:

                            # getting a batch from the memory buffer
                            try:
                                x_buff, y_buff, _, _, _, _, zero_based_train_task_id, _ = next(replay_set_iter)
                            except StopIteration:
                                replay_set_iter = iter(replay_set_data_loader)
                                x_buff, y_buff, _, _, _, _, zero_based_train_task_id, _ = next(replay_set_iter)

                            # moving experiences and casting
                            x_buff = x_buff.to(opts['device'])
                            y_buff = y_buff.to(torch.float32).to(opts['device'])
                            zero_based_train_task_id = zero_based_train_task_id.to(opts['device'])

                            # prediction on selected experiences (they might belong to different tasks)
                            c_pred_buff, c_embs_buff, o = net(x_buff)
                            o = o.gather(1, zero_based_train_task_id.unsqueeze(-1)).squeeze(-1)

                            # loss evaluation on the retrieved experiences (from raw outputs)
                            replay_loss = torch.nn.functional.binary_cross_entropy_with_logits(o, y_buff, reduction='mean')
                            loss += opts['replay_lambda'] * torch.nan_to_num(replay_loss)

                            replay_loss = replay_loss.item()


                    # possibly storing the current example(s) to the memory buffer
                    added_something = False

                    for i, abs_j in enumerate(abs_idx):
                        added_something = train_set.buffer_sample(abs_j.item(), torch.zeros(c_pred.shape[1], dtype=torch.float).to(opts['device']), opts['balance']) or added_something

                    # if the buffer changed, re-create the data sampler
                    if added_something:
                        replay_set_data_loader = \
                            DataLoader(train_set,
                                       batch_size=opts['batch'],
                                       sampler=SubsetRandomSampler(train_set.get_buffered_sample_indices()))
                        replay_set_iter = iter(replay_set_data_loader)

                # running average
                avg_loss = (0.9 * avg_loss + 0.1 * loss) if n > 0 else loss

                # printing
                n += opts['batch']
                n = min(n, num_training_examples)
                if (n // opts['batch']) % opts['print_every'] == 0:
                    print("[TrainTask: " + str(train_task_id + 1) + "/" + str(len(train_sets)) +
                          ", Epoch: " + str(task_epoch + 1) + "/" + str(num_task_epochs) +
                          ", Sample: " + str(n) + "/" + str(num_training_examples) + "]" +
                          " Loss: {0:.4f}, AvgLoss: {1:.4f}".format(loss.item(), avg_loss.item()))
                    print("Class loss: {:.4f}, Concept loss {:.4f}, Concept pol: {:.4f}, Mask pol: {:.4f}, ".format(
                        cls_loss,
                        concept_loss,
                        concept_pol_loss,
                        mask_pol_loss
                    ) +
                          "Triplet loss (batch): {:.4f}, Triplet loss (buff): {:.4f}, Replay loss: {:.4f}".format(
                              triplet_loss_batch,
                                    triplet_loss_buffer,
                                    replay_loss
                          ))

                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # On epoch end, store the last loss values:
            metrics_train['loss'].append(loss.item())
            metrics_train['cls_loss'].append(cls_loss)
            metrics_train['concept_loss'].append(concept_loss)
            metrics_train['concept_pol_loss'].append(concept_pol_loss)
            metrics_train['mask_pol_loss'].append(mask_pol_loss)
            metrics_train['triplet_loss_batch'].append(triplet_loss_batch)
            metrics_train['triplet_loss_buffer'].append(triplet_loss_buffer)
            metrics_train['replay_loss'].append(replay_loss)

            # On epoch end, update buffer representations:
            with torch.no_grad():
                if opts['replay_buffer'] > 0:
                    for b in range(0, len(train_set.buffered_indices), opts['batch']):
                        if opts['model'] == 'mlp':
                            x_buff_batch[:,:] = 0.
                        else:
                            x_buff_batch[:,:,:,:] = 0.
                        for i, idx in enumerate(train_set.buffered_indices[b:b+opts['batch']]):
                            x_buff_batch[i], _, _, _, _, _, _, _ = train_set[idx]

                        actual_imgs = len(train_set.buffered_indices[b:b+opts['batch']])
                        c_pred, _, _ = net(x_buff_batch[:actual_imgs])
                        c_pred.cpu()

                        for i, idx in enumerate(train_set.buffered_indices[b:b+opts['batch']]):
                            if opts['store_fuzzy']:
                                train_set.update_representation(idx, c_pred[i,:])
                            else:
                                train_set.update_representation(idx, torch.gt(c_pred[i, :], 0.5))


        # standardizing data for evaluation purposes
        eval_task_id = train_task_id if opts['train'] != 'joint' else num_tasks - 1
        backup_train_set_transform = None
        acc_per_task = None
        concept_vectors = None
        compute_metrics_on_train_data_too = opts['compute_training_metrics']

        # evaluation (validation set -with decision threshold tuning-, training set, test set)
        for eval_set, metrics in zip([val_set, train_set, test_set],
                                     [metrics_val, metrics_train, metrics_test]):

            # temporarily replacing transformations for training data
            if eval_set == train_set:
                backup_train_set_transform = train_set.transform
                train_set.transform = val_set.transform

            # computing accuracies
            if compute_metrics_on_train_data_too or eval_set != train_set:
                print("Computing metrics on " + metrics['name'] + " data...")
                acc_per_task, concept_vectors = compute_matrices(net if opts['train'] != 'independent' else independent_nets,
                                                  eval_set,
                                                  batch_size=32,
                                                  device=opts['device'],
                                                  tune_decision_thresholds=eval_set == val_set,
                                                  tune_last_task_only=False)
                # tune_decision_thresholds=(eval_set == val_set) and opts['train'] != 'continual_online',
                # tune_last_task_only=opts['train'] != 'joint')

                for k in concept_vectors[eval_task_id].keys():
                    if k not in extended_concept_vectors:
                        extended_concept_vectors[k] = concept_vectors[eval_task_id][k]
                    else:
                        extended_concept_vectors[k] = np.concatenate([extended_concept_vectors[k], concept_vectors[eval_task_id][k]], axis=0)


            else:
                acc_per_task = [-1.] * len(acc_per_task)  # acc_per_task was populated during the val-set evaluation

            # updating accuracy matrix
            metrics['acc_matrix'][eval_task_id, :] = torch.tensor(acc_per_task)

            # fixing the 'joint' case (in a nutshell: repeating the same results many times to fill up the matrix)
            if opts['train'] == 'joint':
                metrics['acc_matrix'][0:-1, :] = metrics['acc_matrix'][-1, :]

            # limiting the accuracy matrix to the current training task
            acc_matrix_so_far = metrics['acc_matrix'][0:eval_task_id + 1, 0:eval_task_id + 1]

            # updating all the metrics
            metrics['avg_accuracy'][eval_task_id] = avg_accuracy(acc_matrix_so_far)
            metrics['avg_forgetting'][eval_task_id] = avg_forgetting(acc_matrix_so_far)
            metrics['backward_transfer'][eval_task_id] = backward_transfer(acc_matrix_so_far)
            metrics['forward_transfer'][eval_task_id] = forward_transfer(acc_matrix_so_far)

            if concept_vectors is None:
                metrics['cas'][eval_task_id] = 0
                metrics['tas'][eval_task_id] = 0
                metrics['cas_extended'][eval_task_id] = 0
                metrics['tas_extended'][eval_task_id] = 0
            else:
                c_pred_for_cas = concept_vectors[eval_task_id]['c_embs']
                c_test_for_cas = concept_vectors[eval_task_id]['c_true']
                c_pred_for_cas = np.broadcast_to(np.expand_dims(c_pred_for_cas,axis=1),
                                                 (c_pred_for_cas.shape[0], c_test_for_cas.shape[1],c_pred_for_cas.shape[1])) # N x A x C
                #c_pred_for_cas = c_pred_for_cas.reshape((c_pred_for_cas.shape[0], -1)) # N x AC

                metrics['cas'][eval_task_id], metrics['tas'][eval_task_id] = concept_alignment_score(
                    c_vec=c_pred_for_cas,
                    c_test=c_test_for_cas,
                    y_test=concept_vectors[eval_task_id]['pseudo_y'],
                    step=5)

                c_pred_for_cas = extended_concept_vectors['c_embs']
                c_test_for_cas = extended_concept_vectors['c_true']
                c_pred_for_cas = np.broadcast_to(np.expand_dims(c_pred_for_cas, axis=1),
                                                 (c_pred_for_cas.shape[0], c_test_for_cas.shape[1],
                                                  c_pred_for_cas.shape[1]))  # N x A x C
                #c_pred_for_cas = c_pred_for_cas.reshape((c_pred_for_cas.shape[0], -1))  # N x AC

                metrics['cas_extended'][eval_task_id], metrics['tas_extended'][eval_task_id] = concept_alignment_score(
                    c_vec=c_pred_for_cas,
                    c_test=c_test_for_cas,
                    y_test=extended_concept_vectors['pseudo_y'],
                    step=5)

                # OLD
                #metrics['cas'][eval_task_id], metrics['tas'][eval_task_id] = concept_alignment_score(c_vec=concept_vectors[eval_task_id]['c_embs'],
                #                                                       c_test=concept_vectors[eval_task_id]['c_true'],
                #                                                       y_test=concept_vectors[eval_task_id]['pseudo_y'],
                #                                                       step=5)
                #metrics['extended_cas'][eval_task_id], metrics['extended_tas'][eval_task_id] = concept_alignment_score(
                #    c_vec=extended_concept_vectors['c_embs'],
                #    c_test=extended_concept_vectors['c_true'],
                #    y_test=extended_concept_vectors['pseudo_y'],
                #    step=5)
                # END OLD

                true_concept_len = concept_vectors[eval_task_id]['c_true'].shape[1]
                if opts['correlate_each_task']:
                    #_, pp, pt = pearson_corr(**concept_vectors[eval_task_id])
                    #pp['data'] = pp['data'].tolist()
                    #pt['data'] = pt['data'].tolist()

                    #metrics['concept_correlation_pearson_pp_continual'].append(pp)
                    #metrics['concept_correlation_pearson_pt_continual'].append(pt)

                    _, pp, pt = matthews_corr(**concept_vectors[eval_task_id])
                    pp['data'] = pp['data'].tolist()
                    pt['data'] = pt['data'].tolist()

                    metrics['concept_correlation_phi_pp_continual'].append(pp)
                    metrics['concept_correlation_phi_pt_continual'].append(pt)

                    #_, _, pt = raw_counts(**concept_vectors[eval_task_id])
                    #pt['data'] = pt['data'].tolist()
                    #metrics['counts_pt_continual'].append(pt)

                    #_, pp, pt = pearson_corr(**extended_concept_vectors)
                    #pp['data'] = pp['data'].tolist()
                    #pt['data'] = pt['data'].tolist()

                    #metrics['concept_correlation_pearson_pp_continual_extended'].append(pp)
                    #metrics['concept_correlation_pearson_pt_continual_extended'].append(pt)

                    _, pp, pt = matthews_corr(**extended_concept_vectors)
                    pp['data'] = pp['data'].tolist()
                    pt['data'] = pt['data'].tolist()

                    metrics['concept_correlation_phi_pp_continual_extended'].append(pp)
                    metrics['concept_correlation_phi_pt_continual_extended'].append(pt)

                    #_, _, pt = raw_counts(**extended_concept_vectors)
                    #pt['data'] = pt['data'].tolist()
                    #metrics['counts_pt_continual_extended'].append(pt)

                if eval_task_id == num_tasks - 1:
                    #(metrics['concept_correlation_pearson_tt'],
                    # metrics['concept_correlation_pearson_pp'],
                    # metrics['concept_correlation_pearson_pt']) = pearson_corr(**extended_concept_vectors)
                    (metrics['concept_correlation_phi_tt'],
                     metrics['concept_correlation_phi_pp'],
                     metrics['concept_correlation_phi_pt']) = matthews_corr(**extended_concept_vectors)
                    #(metrics['counts_t'],
                    # metrics['counts_p'],
                    # metrics['counts_pt']) = raw_counts(**extended_concept_vectors)

            # fixing the 'joint' case (in a nutshell: repeating the same results many times to fill up the arrays)
            if opts['train'] == 'joint':
                metrics['avg_accuracy'][0:-1] = [metrics['avg_accuracy'][-1]] * (num_tasks - 1)
                metrics['avg_forgetting'][0:-1] = [metrics['avg_forgetting'][-1]] * (num_tasks - 1)
                metrics['backward_transfer'][0:-1] = [metrics['backward_transfer'][-1]] * (num_tasks - 1)
                metrics['forward_transfer'][0:-1] = [metrics['forward_transfer'][-1]] * (num_tasks - 1)
                metrics['cas'][0:-1] = [metrics['cas'][-1]] * (num_tasks - 1)
                metrics['tas'][0:-1] = [metrics['tas'][-1]] * (num_tasks - 1)
                metrics['cas_extended'][0:-1] = [metrics['cas_extended'][-1]] * (num_tasks - 1)
                metrics['tas_extended'][0:-1] = [metrics['tas_extended'][-1]] * (num_tasks - 1)

            # printing
            print_metrics(metrics, train_task_id + 1)

            # restoring transformations for training data
            if eval_set == train_set:
                train_set.transform = backup_train_set_transform

        # freeing GPU memory, if the network was targeted to GPU
        if opts['train'] == 'independent':
            net.cpu()

    # converting Pytorch accuracy matrix to list of lists (that is JSON serializable, useful for saving operations)
    metrics_train['acc_matrix'] = metrics_train['acc_matrix'].numpy().tolist()
    metrics_val['acc_matrix'] = metrics_val['acc_matrix'].numpy().tolist()
    metrics_test['acc_matrix'] = metrics_test['acc_matrix'].numpy().tolist()

    if opts['correlate_each_task']:
        p_label = metrics_test['concept_correlation_phi_pt_continual'][0]['x_label']
        t_label = metrics_test['concept_correlation_phi_pt_continual'][0]['y_label']
    else:
        p_label = metrics_test['concept_correlation_phi_pt']['x_label']
        t_label = metrics_test['concept_correlation_phi_pt']['y_label']

    #metrics_train['concept_correlation_pearson_tt'] = metrics_train['concept_correlation_pearson_tt']['data'].tolist()
    #metrics_train['concept_correlation_pearson_pp'] = metrics_train['concept_correlation_pearson_pp']['data'].tolist()
    #metrics_train['concept_correlation_pearson_pt'] = metrics_train['concept_correlation_pearson_pt']['data'].tolist()
    #metrics_val['concept_correlation_pearson_tt'] = metrics_val['concept_correlation_pearson_tt']['data'].tolist()
    #metrics_val['concept_correlation_pearson_pp'] = metrics_val['concept_correlation_pearson_pp']['data'].tolist()
    #metrics_val['concept_correlation_pearson_pt'] = metrics_val['concept_correlation_pearson_pt']['data'].tolist()
    #metrics_test['concept_correlation_pearson_tt'] = metrics_test['concept_correlation_pearson_tt']['data'].tolist()
    #metrics_test['concept_correlation_pearson_pp'] = metrics_test['concept_correlation_pearson_pp']['data'].tolist()
    #metrics_test['concept_correlation_pearson_pt'] = metrics_test['concept_correlation_pearson_pt']['data'].tolist()

    metrics_train['concept_correlation_phi_tt'] = metrics_train['concept_correlation_phi_tt']['data'].tolist()
    metrics_train['concept_correlation_phi_pp'] = metrics_train['concept_correlation_phi_pp']['data'].tolist()
    metrics_train['concept_correlation_phi_pt'] = metrics_train['concept_correlation_phi_pt']['data'].tolist()
    metrics_val['concept_correlation_phi_tt'] = metrics_val['concept_correlation_phi_tt']['data'].tolist()
    metrics_val['concept_correlation_phi_pp'] = metrics_val['concept_correlation_phi_pp']['data'].tolist()
    metrics_val['concept_correlation_phi_pt'] = metrics_val['concept_correlation_phi_pt']['data'].tolist()
    metrics_test['concept_correlation_phi_tt'] = metrics_test['concept_correlation_phi_tt']['data'].tolist()
    metrics_test['concept_correlation_phi_pp'] = metrics_test['concept_correlation_phi_pp']['data'].tolist()
    metrics_test['concept_correlation_phi_pt'] = metrics_test['concept_correlation_phi_pt']['data'].tolist()

    #metrics_train['counts_t'] = metrics_train['counts_t']['data'].tolist()
    #metrics_train['counts_p'] = metrics_train['counts_p']['data'].tolist()
    #metrics_train['counts_pt'] = metrics_train['counts_pt']['data'].tolist()
    #metrics_val['counts_t'] = metrics_val['counts_t']['data'].tolist()
    #metrics_val['counts_p'] = metrics_val['counts_p']['data'].tolist()
    #metrics_val['counts_pt'] = metrics_val['counts_pt']['data'].tolist()
    #metrics_test['counts_t'] = metrics_test['counts_t']['data'].tolist()
    #metrics_test['counts_p'] = metrics_test['counts_p']['data'].tolist()
    #metrics_test['counts_pt'] = metrics_test['counts_pt']['data'].tolist()




    return metrics_train, metrics_val, metrics_test, p_label, t_label