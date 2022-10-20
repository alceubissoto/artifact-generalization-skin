import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer
from dataset_loader import CSVDatasetWithName, CSVDataset
#from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import confusion_matrix, roc_auc_score
from data.skin_dataset import get_transform_skin
import pickle

class AugmentOnTest:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n * len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i // self.n]

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    scaler = torch.cuda.amp.GradScaler()
    all_preds = []
    all_labels = []
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            with torch.cuda.amp.autocast(): 
                outputs = model(x)
                loss_main = loss_computer.loss(outputs, y, g, is_training)

            all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
            all_labels += list(y.cpu().data.numpy())

            if is_training:
                optimizer.zero_grad()
                scaler.scale(loss_main).backward()
                scaler.step(optimizer)
                scaler.update()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()


        # Calculate multiclass AUC
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_preds[:, 1])
        print('Epoch: ' + str(epoch) + ' -- AUC: ', str(auc))

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
        return auc

def run_eval(epoch, model, loader, args, test_aug=False):
    
    model.eval()
    prog_bar_loader = loader

    all_preds = []
    all_labels = []
    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(prog_bar_loader):
            x, y, _ = batch
            x = x.to("cuda")
            y = y.to("cuda")
            outputs = model(x)
            if not test_aug:
                all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
                all_labels += list(y.cpu().data.numpy())
            else:
                all_preds += list([np.mean(F.softmax(outputs, dim=1).cpu().data.numpy(), axis=0)])
                all_labels += list([y.cpu().data.numpy()[0]])
        # Calculate multiclass AUC
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_preds[:, 1])

        cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        acc = np.trace(cmn) / cmn.shape[0]
        print('Epoch: ' + str(epoch) + ' -- AUC: ' + str(auc) + ' -- ACC: ' + str(acc) )
        return auc, acc

def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, fs_observer, wandb, epoch_offset):
    print("fffss observer", fs_observer)
    model = model.cuda()
    CHECKPOINTS_DIR = os.path.join(fs_observer, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_best')
    LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_last')
    os.makedirs(CHECKPOINTS_DIR)
    patience_count = 0
    wandb.log({'args':args}, commit=True)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)
    else:
        scheduler = None

    best_val_loss = 100000
    best_val_auc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        train_auc = run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args, 
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        val_auc = run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer, 
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        #if dataset['test_data'] is not None:
        #    test_loss_computer = LossComputer(
        #        criterion,
        #        is_robust=args.robust,
        #        dataset=dataset['test_data'],
        #        step_size=args.robust_step_size,
        #        alpha=args.alpha)
        #    run_epoch(
        #        epoch, model, optimizer,
        #        dataset['test_loader'],
        #        test_loss_computer,
        #        None, test_csv_logger, args,
        #        is_training=False)


        metrics_comet = {'epoch':epoch, 'train/loss': train_loss_computer.avg_actual_loss, 'train/max_group_loss': max(train_loss_computer.exp_avg_loss), 'train/auc': train_auc, 'val/loss': val_loss_computer.avg_actual_loss, 'val/max_group_loss': max(val_loss_computer.avg_group_loss), 'val/auc': val_auc}
        wandb.log(metrics_comet, commit=True)

        patience_count += 1
        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        #if epoch < args.save_step:
        #    torch.save(model,  LAST_MODEL_PATH + str(epoch) + '.pth')

        if args.save_last:
            torch.save(model,  LAST_MODEL_PATH + '.pth')

        if args.save_best:
            if args.robust or args.reweight_groups:
                #curr_val_loss = max(val_loss_computer.avg_group_loss)
                curr_val_loss = val_loss_computer.avg_actual_loss
            else:
                curr_val_loss = val_loss_computer.avg_actual_loss
            logger.write(f'Current validation loss: {curr_val_loss}\n')
            #if curr_val_loss < best_val_loss:
            if val_auc > best_val_auc:
                #best_val_loss = curr_val_loss
                best_val_auc = val_auc
                torch.save(model,  BEST_MODEL_PATH + '.pth')
                logger.write(f'Best model saved at epoch {epoch}\n')
                patience_count = 0

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')

        # Early Stopping
        if patience_count > args.patience:
            break

    # RUN TESTS
    print("Performing tests. Loading model at {}".format(BEST_MODEL_PATH))
    model = torch.load('{}.pth'.format(BEST_MODEL_PATH))
    

    # Test for Skin
    if args.dataset == 'Skin':
        test_ds_atlas_clin = CSVDataset('/artifact-generalization-skin/datasets/edraAtlas', '/artifact-generalization-skin/datasets/edraAtlas/atlas-clinical-all.csv', 'image', 'label', transform=get_transform_skin(False), add_extension='.jpg')
        test_ds_atlas_derm = CSVDataset('/artifact-generalization-skin/datasets/edraAtlas', '/artifact-generalization-skin/datasets/edraAtlas/atlas-dermato-all.csv', 'image', 'label',transform=get_transform_skin(False), add_extension='.jpg')
        test_ds_ph2 = CSVDataset('/artifact-generalization-skin/datasets/ph2images/', '/artifact-generalization-skin/datasets/ph2images/ph2.csv', 'image', 'label',transform=get_transform_skin(False), add_extension='.jpg')
        test_ds_padufes = CSVDataset('/artifact-generalization-skin/datasets/pad-ufes/', '/artifact-generalization-skin/datasets/pad-ufes/padufes-test-wocarc.csv', 'img_id', 'label',transform=get_transform_skin(False), add_extension=None)

        shuffle = False
        data_sampler = None
        num_workers = 8
        REPLICAS = 50
        dataloaders_atlas_dermato = {
            'val': DataLoader(AugmentOnTest(test_ds_atlas_derm, REPLICAS), batch_size=REPLICAS,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }
        dataloaders_atlas_clin = {
            'val': DataLoader(AugmentOnTest(test_ds_atlas_clin, REPLICAS), batch_size=REPLICAS,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }
        dataloaders_ph2 = {
            'val': DataLoader(AugmentOnTest(test_ds_ph2, REPLICAS), batch_size=REPLICAS,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }
        dataloaders_padufes = {
            'val': DataLoader(AugmentOnTest(test_ds_padufes, REPLICAS), batch_size=REPLICAS,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }

        atlas_dermato_auc = run_eval(
                epoch, model,
                dataloaders_atlas_dermato['val'],
                args)
        atlas_clin_auc = run_eval(
                epoch, model,
                dataloaders_atlas_clin['val'],
                args)
        ph2_auc = run_eval(
                epoch, model,
                dataloaders_ph2['val'],
                args)
        padufes_auc = run_eval(
                epoch, model,
                dataloaders_padufes['val'],
                args)
        metrics_ood = {'atlas_dermato/auc': atlas_dermato_auc,  'atlas_clin/auc': atlas_clin_auc,  'ph2/auc': ph2_auc, 'padufes/auc': padufes_auc}
        print(metrics_ood)
        wandb.log(metrics_ood, commit=True)
        
        test_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['test_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)

        test_loader = DataLoader(AugmentOnTest(dataset['test_data'], REPLICAS), batch_size=REPLICAS, shuffle=shuffle, num_workers=num_workers, sampler=data_sampler, pin_memory=True)

        test_auc = run_epoch(
            epoch, model, optimizer,
            test_loader,
            test_loss_computer,
            None, test_csv_logger, args,
            is_training=False) 
        metrics_test = {'test/auc': test_auc}
        print(metrics_test)
        wandb.log(metrics_test, commit=True)

