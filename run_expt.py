from sacred import Experiment
from sacred.observers import FileStorageObserver, TelegramObserver
ex = Experiment()
import wandb
import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import pretrainedmodels as ptm
from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train
from sacred.observers import RunObserver
import numpy as np

class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'

class SetID(RunObserver):
    priority = 50  # very high priority

    def __init__(self, custom_id):
        self.custom_id = custom_id

    def started_event(self, ex_info, command, host_info, start_time,  config, meta_info, _id):
        return self.custom_id    # started_event should returns the _id

telegram_file = 'telegram.json'
if os.path.isfile(telegram_file):
    telegram_obs = TelegramObserver.from_config(telegram_file)
    ex.observers.append(telegram_obs)
fs_observer = FileStorageObserver.create('results-comet')


def save_images(dataset, to, n=32):
    for i in range(n):
        img_path = os.path.join(to, 'img_{}.png'.format(i))
        save_image(dataset[i][0], img_path)

@ex.config
def config():
    # Settings
    cfg = {
    "dataset": None,
    "shift_type": None,
    "target_name": None,
    "confounder_names": [],
    "resume": False,
    "minority_fraction": None,
    "imbalance_ratio": None,
    "fraction": 1.0,
    "root_dir": None,
    "reweight_groups": False,
    "augment_data": False,
    "val_fraction": 0.1,
    "train_csv": None,
    "val_csv": None,
    "test_csv": None,
    "robust":False,
    "alpha": 0.2,
    "generalization_adjustment":"0.0",
    "automatic_adjustment": False,
    "robust_step_size": 0.01,
    "use_normalized_loss": False,
    "btl": False,
    "hinge": False,
    "model": 'inceptionv4',
    "train_from_scratch": False,
    "n_epochs": 100,
    "batch_size": 32,
    "lr": 0.001,
    "scheduler": False,
    "weight_decay": 5e-5,
    "gamma": 0.1,
    "minimum_variational_weight": 0,
    "seed": None,
    "show_progress": False,
    "log_dir": './logs',
    "log_every": 500,
    "save_step": 5,
    "save_best": True,
    "save_last": True,
    "patience": 22,
    "exp_name": 'tmp',
    "exp_desc": None,
    "sc1": None,
    "sc2": None,
    "square_size": None,
    "color_noise": None,
    "random_square_position": None,
    "bias_factor": None,
    "ln": None,
    "test_noise": None}
    ex.observers.append(fs_observer)
    ex.observers.append(SetID(cfg['exp_name']))

@ex.automain
def main(cfg):

    args = DictX(cfg)
    #wandb.init(name=args.exp_name, project='groupDRO_positionenvs', entity='alceubissoto', save_code=True)
    wandb.init(mode='disabled')
    args.generalization_adjustment = str(args.generalization_adjustment)

    #wandb logging
    config_w = wandb.config
    config_w.train_root=args.root_dir
    config_w.val_root=args.root_dir
    config_w.train_csv=args.train_csv
    config_w.val_csv=args.val_csv
    config_w.model=args.model
    config_w.learning_rate=args.lr
    config_w.batch_size=args.batch_size
    config_w.exp_desc=args.exp_desc

    # BERT-specific configs copied over from run_glue.py
    if args.model == 'bert':
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)
    if args.seed is None:
        config_w.seed = np.random.randint(0,10000)
    else:
        config_w.seed = args.seed
    
    set_seed(config_w.seed)

    log_args(args, logger)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    pretrained = not args.train_from_scratch
    if resume:
        model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'inceptionv4':
        model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
    elif args.model == 'bert':
        assert args.dataset == 'MultiNLI'

        from pytorch_transformers import BertConfig, BertForSequenceClassification
        config_class = BertConfig
        model_class = BertForSequenceClassification

        config = config_class.from_pretrained(
            'bert-base-uncased',
            num_labels=3,
            finetuning_task='mnli')
        model = model_class.from_pretrained(
            'bert-base-uncased',
            from_tf=False,
            config=config)
    else:
        raise ValueError('Model not recognized.')


    logger.flush()


    ## Define the objective
    if args.hinge:
        assert args.dataset in ['CelebA', 'CUB'] # Only supports binary
        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)
    print("fsobserver: ", fs_observer.dir)
    train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, fs_observer.dir, wandb, epoch_offset=epoch_offset)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio



if __name__=='__main__':
    main()

