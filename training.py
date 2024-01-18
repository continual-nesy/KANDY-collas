import torch
import copy
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import TaskOrganizedDataset
from utils import (compute_accuracies, avg_accuracy, avg_forgetting, forward_transfer, backward_transfer, print_metrics,
                   HammingDistance, MaskedTripletMarginLoss)

import pytorch_metric_learning.losses, pytorch_metric_learning.miners
# import cem.metrics.accs
# TODO: cas and oracle require tensorflow because they build keras models...Reimplement them in torch?
#       niching uses predict_proba() (it's a sklearn model)
from metrics import concept_alignment_score, niche_impurity_score, oracle_impurity_score

import random

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
        'concept_avg_accuracy': [-1.] * num_tasks,
        'cas': [-1.] * num_tasks,
        'ois': [-1.] * num_tasks,
        'nis': [-1.] * num_tasks
    }
    metrics_val = copy.deepcopy(metrics_train)
    metrics_val['name'] = 'val'
    metrics_test = copy.deepcopy(metrics_train)
    metrics_test['name'] = 'test'
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
                loss = torch.nn.functional.binary_cross_entropy_with_logits(o, y, reduction='mean') # Task loss.

                positive_samples = torch.nonzero(y == 1).reshape((-1,))

                if opts['concept_lambda'] > 0. and opts['min_pos_concepts'] > 0:
                    loss += opts['concept_lambda'] * \
                            torch.mean(
                                torch.clamp(torch.sum(c_pred[positive_samples]) - float(opts['min_pos_concepts']), 0))

                if opts['concept_polarization_lambda'] > 0.:
                    loss += -opts['concept_polarization_lambda'] * \
                            torch.nn.functional.l1_loss(c_pred,
                                                        zero_five[:c_pred.shape[0],:],
                                                        reduction="mean")


                if opts['mask_polarization_lambda'] > 0. and opts['use_mask'] == 'fuzzy':
                    mask, _ = distance_fn.soft_intersection(c_pred[positive_samples])
                    loss += -opts['mask_polarization_lambda'] * \
                            torch.nn.functional.l1_loss(mask, zero_five[:mask.shape[0],:],
                                                        reduction="mean")

                # Hamming loss:
                if opts['triplet_lambda'] > 0.:
                    if opts['batch'] > 3:
                        indices_tuple = mining_fn(c_pred, eq_classes)
                        triplet_loss = hamming_loss_fn(c_pred, eq_classes, indices_tuple=indices_tuple,
                                                        positives=c_pred[positive_samples])

#########################################################################################################################################
# TODO: Qualcosa non quadra da qui
                    if opts['replay_buffer'] > 2:
                        if len(train_set.buffered_indices) > 0 and \
                            train_task_id in train_set.task2buffered_positives and \
                            train_task_id in train_set.task2buffered_negatives and \
                            positive_samples.shape[0] > 0 and \
                            len(train_set.task2buffered_positives[train_task_id]) > 0 and \
                            len(train_set.task2buffered_negatives[train_task_id]) > 0:


                            triplet_p = torch.zeros(c_pred[positive_samples].shape, dtype=torch.float)
                            triplet_n = torch.zeros(c_pred[positive_samples].shape, dtype=torch.float)

                            buffered_p = torch.zeros(c_pred.shape, dtype=torch.float) # Positives in buffer for mask re-computation

                            for i in range(c_pred[positive_samples].shape[0]):
                                _, _, _, _, triplet_p[i,:], _, _, _ = train_set[random.choice(train_set.task2buffered_positives[train_task_id])]
                                _, _, _, _, triplet_n[i,:], _, _, _ = train_set[random.choice(train_set.task2buffered_negatives[train_task_id])]
                                buffered_p[i, :] = triplet_p[i,:]

                            triplet_p = triplet_p.to(opts['device']).detach()
                            triplet_n = triplet_n.to(opts['device']).detach()
                            buffered_p = buffered_p.to(opts['device']).detach()

                            print(triplet_p)
                            print(triplet_n)
                            ERROR # TODO: Perché alcune entry sono zero, anche se prese da task2buffered_*???


                            if opts['use_mask'] == 'crisp':
                                _, mask = distance_fn.soft_intersection(buffered_p)
                            elif opts['use_mask'] == 'fuzzy':
                                mask, _ = distance_fn.soft_intersection(buffered_p)
                            else:
                                mask = None

                            ap = distance_fn.hamming_distance_01_masked(c_pred[positive_samples], triplet_p, mask)
                            an = distance_fn.hamming_distance_01_masked(c_pred[positive_samples], triplet_n, mask)


                            current_margins = distance_fn.margin(ap, an)
                            violation = current_margins + hamming_loss_fn.margin


                            if hamming_loss_fn.smooth_loss:
                                loss_mat = torch.nn.functional.softplus(violation)
                            else:
                                loss_mat = torch.nn.functional.relu(violation)

                            triplet_loss += torch.mean(loss_mat[torch.gt(loss_mat, 0.)]) # AvgNonZero reduction

# A qui
#########################################################################################################################################

                            # possibly storing the current example(s) to the memory buffer
                            added_something = False

                            for i, abs_j in enumerate(abs_idx):
                                if opts['store_fuzzy']:
                                    added_something = train_set.buffer_sample(abs_j.item(), c_pred[i, :],
                                                                              opts['balance']) or added_something
                                else:
                                    added_something = train_set.buffer_sample(abs_j.item(),
                                                                              torch.gt(c_pred[i, :], 0.5).to(torch.int),
                                                                              opts['balance']) or added_something

                    if opts['triplet_lambda'] > 0. and opts['replay_buffer'] > 2:
                        loss += opts['triplet_lambda'] + triplet_loss / 2.
                    else:
                        loss += opts['triplet_lambda'] + triplet_loss

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
                            loss += opts['replay_lambda'] * \
                                torch.nn.functional.binary_cross_entropy_with_logits(o, y_buff, reduction='mean')

                    # possibly storing the current example(s) to the memory buffer
                    added_something = False

                    for i, abs_j in enumerate(abs_idx):
                        if opts['store_fuzzy']:
                            added_something = train_set.buffer_sample(abs_j.item(), c_pred[i,:], opts['balance']) or added_something
                        else:
                            added_something = train_set.buffer_sample(abs_j.item(), torch.gt(c_pred[i, :], 0.5).to(torch.int),
                                                                      opts['balance']) or added_something

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

                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # standardizing data for evaluation purposes
        eval_task_id = train_task_id if opts['train'] != 'joint' else num_tasks - 1
        backup_train_set_transform = None
        acc_per_task = None
        compute_metrics_on_train_data_too = False  # TODO turn this on again (turned off to speed-up experiments)

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
                acc_per_task = compute_accuracies(net if opts['train'] != 'independent' else independent_nets,
                                                  eval_set,
                                                  batch_size=32,
                                                  device=opts['device'],
                                                  tune_decision_thresholds=eval_set == val_set,
                                                  tune_last_task_only=False)
                # tune_decision_thresholds=(eval_set == val_set) and opts['train'] != 'continual_online',
                # tune_last_task_only=opts['train'] != 'joint')
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

            metrics['concept_avg_accuracy'][eval_task_id] = 0
            metrics['cas'][eval_task_id] = 0
            metrics['ois'][eval_task_id] = 0
            metrics['nis'][eval_task_id] = 0

            # fixing the 'joint' case (in a nutshell: repeating the same results many times to fill up the arrays)
            if opts['train'] == 'joint':
                metrics['avg_accuracy'][0:-1] = [metrics['avg_accuracy'][-1]] * (num_tasks - 1)
                metrics['avg_forgetting'][0:-1] = [metrics['avg_forgetting'][-1]] * (num_tasks - 1)
                metrics['backward_transfer'][0:-1] = [metrics['backward_transfer'][-1]] * (num_tasks - 1)
                metrics['forward_transfer'][0:-1] = [metrics['forward_transfer'][-1]] * (num_tasks - 1)
                metrics['concept_avg_accuracy'][0:-1] = [metrics['concept_avg_accuracy'][-1]] * (num_tasks - 1)
                metrics['cas'][0:-1] = [metrics['cas'][-1]] * (num_tasks - 1)
                metrics['ois'][0:-1] = [metrics['ois'][-1]] * (num_tasks - 1)
                metrics['nis'][0:-1] = [metrics['nis'][-1]] * (num_tasks - 1)

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

    return metrics_train, metrics_val, metrics_test