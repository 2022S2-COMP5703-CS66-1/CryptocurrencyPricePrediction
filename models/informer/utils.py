import torch
import pandas as pd
import numpy as np
import datetime
import copy
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from models.informer.model import Informer
from models.informer.trendloss import TrendLoss


class BTCDataSet(Dataset):

    def __init__(self, array, seq_len, pred_len, label_len, device):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.r_begin = seq_len - label_len
        self.device = device

        self.x = torch.Tensor(array).to(device)
        self.x = self.x.unfold(0, seq_len + pred_len, 1)
        self.x = self.x.transpose(1, 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        window = self.x[index]
        enc_in = window[:self.seq_len]
        dec_in = torch.cat(
            (window[self.r_begin:-self.pred_len], torch.zeros(self.pred_len, self.x.shape[-1]).to(self.device)))
        y = window[-self.pred_len:, :1]

        return enc_in, dec_in, y


class EarlyStopping:

    def __init__(self, tolerance=2):
        self.best = None
        self.counter = 0
        self.tolerance = tolerance

    def __call__(self, criterion):
        if self.best is None:
            self.best = criterion
            return True, True
        if criterion < self.best:
            self.best = criterion
            self.counter = 0
            return True, True
        else:
            self.counter += 1
            return self.counter <= self.tolerance, False


def _pass(criterion):
    return True


class Trainer:

    def __init__(self, data_file_path,
                 conv_trans=False,
                 trend_loss=False,
                 trend_c=0.1,
                 pattern_embedding=False,
                 tanh_position_encoding=False,
                 learning_rate=5e-3,
                 batch_size=128,
                 epoch=5,
                 optim="adamw",
                 weight_decay=.01,
                 betas=(0.9, 0.999),
                 momentum=0.9,
                 early_stopping=True,
                 tolerance=2,
                 lr_decay=0.5,
                 lr_decay_round=50,
                 in_dim=59,
                 out_dim=1,
                 pred_len=168,
                 seq_len=336,
                 label_len=168,
                 n_heads=4,
                 dilation_n_hidden=2,
                 dropout=0.,
                 d_ff=512,
                 d_model=512,
                 conv_trans_kernel_size=8,
                 enc_num=3,
                 dec_num=2,
                 test_begin_date='2022/1/21',
                 train_end_date='2022/1/1',
                 device=torch.device('cuda'),
                 verbose=True,
                 log=True,
                 out_history=True,
                 save_model=True,
                 random_state=0,
                 low_memory=False,
                 model=None):

        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)

        self.params = {"data_file_path": data_file_path,
                       "conv_trans": conv_trans,
                       "trend_loss": trend_loss,
                       "pattern_embedding": pattern_embedding,
                       "tanh_position_encoding": tanh_position_encoding,
                       "learning_rate": learning_rate,
                       "batch_size": batch_size,
                       "epoch": epoch,
                       "optim": optim,
                       "weight_decay": weight_decay,
                       "betas": betas,
                       "momentum": momentum,
                       "early_stopping": early_stopping,
                       "tolerance": tolerance,
                       "lr_decay": lr_decay,
                       "lr_decay_round": lr_decay_round,
                       "in_dim": in_dim,
                       "out_dim": out_dim,
                       "pred_len": pred_len,
                       "seq_len": seq_len,
                       "label_len": label_len,
                       "n_heads": n_heads,
                       "dilation_n_hidden": dilation_n_hidden,
                       "dropout": dropout,
                       "d_ff": d_ff,
                       "d_model": d_model,
                       "conv_trans_kernel_size": conv_trans_kernel_size,
                       "enc_num": enc_num,
                       "dec_num": dec_num,
                       "test_begin_date": test_begin_date,
                       "train_end_date": train_end_date,
                       "device": str(device),
                       "verbose": verbose,
                       "log": log,
                       "out_history": out_history,
                       "save_model": save_model,
                       "random_state": random_state,
                       "low_memory": low_memory,
                       "model": model if model is not None else "new"}

        self.model = Informer(
            conv_trans=conv_trans,
            pattern_embedding=pattern_embedding,
            enc_in_dim=in_dim,
            dec_in_dim=in_dim,
            c_out_dim=out_dim,
            out_len=pred_len,
            n_heads=n_heads,
            dilation_n_hidden=dilation_n_hidden,
            dropout=dropout,
            d_ff=d_ff,
            d_model=d_model,
            conv_trans_kernel_size=conv_trans_kernel_size,
            e_layers=enc_num,
            d_layers=dec_num,
            positional_embedding='tanh' if tanh_position_encoding else 'trigon'
        ).to(device) if model is None else torch.load(model).to(device)

        df = pd.read_csv(data_file_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        train_df = df[df["DateTime"] < train_end_date].drop(["DateTime"], axis=1)
        test_df = df[df["DateTime"] >= test_begin_date].drop(["DateTime"], axis=1)

        self.train_data_set = BTCDataSet(train_df.values, seq_len, pred_len, label_len, device)
        self.test_data_set = BTCDataSet(test_df.values, seq_len, pred_len, label_len, device)

        self.train_data_loader = DataLoader(self.train_data_set, batch_size=batch_size)
        self.test_data_loader = DataLoader(self.test_data_set, batch_size=batch_size)

        self.criterion = TrendLoss(trend_c) if trend_loss else torch.nn.MSELoss()
        self.metirc = torch.nn.MSELoss()
        self.optimizer = None
        if optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=learning_rate,
                                               betas=betas,
                                               weight_decay=weight_decay)
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=learning_rate,
                                             momentum=momentum,
                                             weight_decay=weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                       mode='min',
                                                                       factor=lr_decay,
                                                                       patience=lr_decay_round)

        self.early_stopping = EarlyStopping(tolerance=tolerance) if early_stopping else _pass

        self.epoch = epoch
        self.epoch_history = pd.DataFrame(columns=[f"Train({self.criterion.__class__.__name__})",
                                                   f"Test({self.criterion.__class__.__name__})"],
                                          dtype=object)
        self.step_train_history = []
        self.step_test_history = []
        self.save_model = save_model
        self.log = log
        self.out_history = out_history
        self.verbose = verbose
        self.low_memory = low_memory

    def train(self):
        best_model = None
        for e in range(self.epoch):
            if self.low_memory:
                torch.cuda.empty_cache()
            self.model.train()
            epoch_history = []
            for enc_in, dec_in, y in tqdm(self.train_data_loader):
                self.optimizer.zero_grad()
                yhat = self.model(enc_in, dec_in)
                loss = self.criterion(yhat, y)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(loss)
                epoch_history.append(loss.cpu().detach().item())
            epoch_loss = np.mean(epoch_history)

            test_loss, test_loss_step_history = self.eval()

            self.epoch_history.loc[f"Epoch {e + 1}"] = [epoch_loss, test_loss]
            self.step_train_history += epoch_history
            self.step_test_history += test_loss_step_history

            if self.verbose:
                print(f"Epoch {e + 1}: Train Loss: {epoch_loss} Test Loss: {test_loss}")

            go_on, is_best = self.early_stopping(test_loss)

            if not go_on:
                break

            if is_best:
                best_model = copy.deepcopy(self.model)
        self.model = best_model if best_model is not None else self.model

        time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

        if not os.path.exists(time_stamp):
            os.makedirs(time_stamp)

        if self.save_model:
            torch.save(self.model.cpu(), os.path.join(time_stamp, "Model.pt"))
        if self.log:
            with open(os.path.join(time_stamp, "config.json"), 'w') as f:
                f.write(json.dumps(self.params))
        if self.out_history:
            self.epoch_history.to_csv(os.path.join(time_stamp, "history.csv"))
            with open(os.path.join(time_stamp, "step_history.json"), 'w') as f:
                f.write(json.dumps({"train": self.step_train_history,
                                    "test": self.step_test_history}))

    def __call__(self, *args):
        return self.model(*args)

    def eval(self):
        self.model.eval()
        if self.low_memory:
            torch.cuda.empty_cache()
        history = []
        with torch.no_grad():
            for enc_in, dec_in, y in self.test_data_loader:
                yhat = self.model(enc_in, dec_in)
                loss = self.metirc(yhat, y)
                history.append(loss.cpu().detach().item())
        return np.mean(history), history
