from datetime import datetime, timedelta
import os
import argparse

parser = argparse.ArgumentParser(description='trigger_identification')

parser.add_argument('--max_epochs', default=None,
                    help=' num of epochs to train', type=int)
parser.add_argument('--warmup_epochs', default=None,
                    help=' num of epochs to train', type=int)
parser.add_argument('--gpu', default=None,
                    help='serial numbers of gpu', type=list)
parser.add_argument('--tasks', default=None,
                    help='task list', nargs='+', type=str)

parser.add_argument('--batch_size', default=5,
                    help='batch size', type=int)
parser.add_argument('--test', default=False,
                    help='whether choose test set as dev set', type=bool)
parser.add_argument('--val_type', default='RANDOM',
                    help='type of validation LOEO/RANDOM', type=str)
parser.add_argument('--test_event', default=None,
                    help='index of test event in LOEO setting 0/1/2/3', type=int)
parser.add_argument('--ins', default=None,
                    help='whether to debug the whole programm with mini data', type=int)
args_parser = parser.parse_known_args()[0]


class Args:
    def __init__(self):
        # environment
        self.data_file = '/remote-home/share/social/datasets/Rumor/PHEME/trigger.csv'
        self.preprocess = True # whether to employ refined preprocessing
        self.fix_sentence = False
        self.pretrain_file = '/remote-home/share/social/pretrained/bert-base-cased'
        self.bert_tweet = 'vinai/bertweet-base'
        self.project = 'trigger_identification'  # project name for wandb logging
        self.variable = ''  # which variable to be compared
        self.model_name = 'BertweetUgrnAware'    # model description
        self.save_dir = './save'  # save directory for logging information
        self.result_dir = './result'
        self.ins = '201'           # instance name of remote machine
        self.num_workers = 4  # num of cpus of the remote machine used in dataloader
        self.gpu = [0]

        # model parameters (general)
        self.hidden_size = 300
        # model parameters (model-specific)
        ## encoder: bert
        self.bert_embedding_size = 768
        # hyper-parameters
        self.balance = True
        self.dropout = 0.3
        self.ex_seed = 0
        self.lr = 2e-5
        self.lr_ratio = 4
        self.lr_ratio2 = 2
        self.weight_decay = 0.01
        self.batch_size = 5
        self.max_epochs = 100
        self.warmup_epochs = 9  # num of epochs for only train trigger identification
        self.jump_epochs = 0 # num of epochs for jumping validation to accelerate training
        self.unfreeze_at_epoch = 0 # num of epochs for LM fine-tuning, default as 0 meaning always update LM parameters
        # experiment
        self.tasks = ['trigger', 'verify']
        self.task_weight = {'trigger':1, 'verify':1}  # L= L_t + L_v * task_weight
        self.task_loss_bound = {'trigger':0.65, 'verify':0.8}
        self.tasks_all = ['trigger', 'verify']
        self.split_seed = 10
        self.val_type = 'RANDOM' # validation type: RANDOM/LOEO
        self.test_event = 0
        self.log_every_n_steps = 200
        self.val_check_interval = 200
        self.num_classes = {'trigger': 4, 'verify': 3}
        self.loss_weight = {'trigger': [0.6, 0.8, 1, 1], 'verify': [1,1,1]}
        


args = Args()

# update parameter from parser
for k, v in args_parser.__dict__.items():
    if v is not None:
        if k == 'gpu':
            v = [_ if isinstance(_, int) else int(_) for _ in v]
        setattr(args, k, v)

if args.val_type == 'RANDOM':
    args.warmup_epochs = 1
    args.max_epochs = 80
else:
    args.warmup_epochs = 9
    args.max_epochs = 50


# initialize directory to save checkpoint
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
# create starting time
args.time_now = (datetime.now()+timedelta(hours=8)).strftime('%Y%m%d-%H%M%S') # China timezone
args.time_now = datetime.now().strftime('%Y%m%d-%H%M%S') # default
# modify balance type (only balance training set in LOEO)
args.balance = False if args.val_type != 'LOEO' else args.balance
# create run name
taskinfo = '+'.join(sorted(args.tasks, reverse=True))
val_info = f'{args.val_type}={args.split_seed}' if args.val_type == 'RANDOM' else f'{args.val_type}={args.test_event}'
val_info = f'{val_info}_test' if args.test else val_info
args.val_info = val_info
ex_info = f's{args.split_seed}_w{args.warmup_epochs}_{args.ins}-{args.gpu[0]}'
args.name = f"{args.variable}_{args.model_name}_{taskinfo}_{val_info}_{ex_info}_{args.time_now}"
print(f"run name: {args.name}")
