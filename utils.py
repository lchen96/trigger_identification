import os
import sys
import json

import pandas as pd

import torch

from torchmetrics.functional import accuracy, f1
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from pytorch_lightning.callbacks import BaseFinetuning, Callback
from pytorch_lightning.trainer.states import RunningStage

import wandb


class DisableValBar(TQDMProgressBar):
    '''customized callback to disable validation bar'''

    def __init__(self):
        super().__init__()

    def init_validation_tqdm(self):
        '''diable validation bar, disable = True'''
        has_main_bar = self.main_progress_bar is not None
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=True,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


class FineTune(BaseFinetuning):
    def __init__(self, args):
        super().__init__()
        self._unfreeze_at_epoch = args.unfreeze_at_epoch
        # self._freeze_at_epoch = args.freeze_at_epoch
        # self._val_type = args.val_type

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model.encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` reaches the threshold, pre-trained LM will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.encoder,
                optimizer=optimizer,
                train_bn=True,
            )

class ValEveryNSteps(Callback):
    def __init__(self, args):
        self.every_n_steps = args.val_check_interval
        self.jump_epochs = args.jump_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if trainer.global_step % self.every_n_steps == 0 and trainer.global_step != 0 and trainer.current_epoch>=self.jump_epochs:
            trainer.training = False
            stage = trainer.state.stage
            trainer.state.stage = RunningStage.VALIDATING
            trainer._run_evaluate()
            trainer.state.stage = stage
            trainer.training = True
            trainer.logger_connector._epoch_end_reached = False


class Evaluation:
    def __init__(self, args) -> None:
        self.val_info = args.val_info
        self.model_name = args.model_name
        self.time_now = args.time_now
        self.test = args.test
        self.name = args.name
        self.result_dir = args.result_dir
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.args = args        
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def _compute_metrics(self, outputs, task=None):
        '''evaluate outputs'''
        task_info = f'{task}_' if task else ''
        # handling output in tensor-form
        preds = outputs[f'{task_info}preds']
        targets = outputs[f'{task_info}targets']
        preds_class = torch.argmax(preds, dim=1)
        num_classes = preds.size(1)
        # compute label distribution
        preds_count = pd.Series(preds_class.tolist()).value_counts().to_dict()
        preds_dist = '/'.join([str(preds_count.get(_, 0))
                               for _ in range(num_classes)])
        targets_count = pd.Series(targets.tolist()).value_counts().to_dict()
        targets_dist = '/'.join([str(targets_count.get(_, 0))
                                for _ in range(num_classes)])
        # compute metrics
        acc = accuracy(preds, targets).item()
        maF = f1(preds_class, targets, average='macro',
                 num_classes=num_classes).item()

        metrics = {
            'model_name': self.model_name,
            'val_info': self.val_info,
            'test': self.test,
            'warmup': self.args.warmup_epochs,
            'split': self.args.split_seed,
            'btz':self.batch_size,
            f'{task_info}test_acc': acc,
            f'{task_info}test_maF': maF,
            f'{task_info}preds_dist': preds_dist,
            f'{task_info}targets_dist': targets_dist,
        }
        return metrics

    def _match(self, outputs, df, task=None):
        '''extract instance information'''
        task_info = f'{task}_' if task else ''
        # handling output in tensor-form
        preds = outputs[f'{task_info}preds']
        targets = outputs[f'{task_info}targets']
        preds_class = torch.argmax(preds, dim=1)
        num_classes = preds.size(1)
        if task == 'trigger':
            ids = outputs['mid']
            ids_field = 'mid'
        elif task == 'verify':
            ids = outputs['cid']
            ids_field = 'cid'
        context = df.loc[ids]['content_clean'].tolist()
        correct = (targets == preds_class).tolist()
        df_instance = pd.DataFrame({
            f'{ids_field}': ids,
            'context': context,
            'target': targets.tolist(),
            'preds': preds_class.tolist(),
            'correct': correct
        })
        return df_instance

    def _log(self, fetch_info, metrics, instance, task=None):
        '''record evaluation result'''
        task_info = f'{task}_' if task else ''
        # overall table
        overall_dict = dict( **metrics, **fetch_info)
        print(overall_dict, end='\n\n')
        overall_table = wandb.Table(columns=list(overall_dict.keys()), data=[
                                    list(overall_dict.values())])
        # instance table
        instance_table = wandb.Table(dataframe=instance)
        log_dict = {}
        log_dict[f'{task_info}result'] = overall_table
        # log_dict[f'{task_info}instance'] = instance_table
        save_dict = {}
        save_dict[f'{task_info}result'] = overall_dict
        save_dict[f'{task_info}config'] = self.args.__dict__
        return log_dict, save_dict

    def _save_local(self, save_dict, task):
        result_file = os.path.join(self.result_dir, f'{self.name}_{task}.json')
        with open(result_file, 'w') as f:
            json.dump(save_dict, f, indent=2)

    def evaluate(self, fetch_info, outputs, df_test, task):
        metrics = self._compute_metrics(outputs, task)
        df_instance = self._match(outputs, df_test, task)
        log_info, save_dict = self._log(fetch_info, metrics, df_instance, task)
        self._save_local(save_dict, task)
        return log_info

