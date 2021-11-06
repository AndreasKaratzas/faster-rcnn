
import os
import numpy as np
import pandas as pd


class EliteModel():
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        self.min_tr_loss = np.inf
        self.min_ob_loss = np.inf
        self.min_cl_loss = np.inf

        self.max_ap_p = 0
        self.max_ap_r = 0

        self.curr_overall_score = 0
        self.last_chkpt_overall_score = 0
        self.epoch = 0
        self.epoch_chkpt = 0
    
    def calculate_metrics(self, epoch: int):
        self.epoch = epoch

        df = pd.read_csv(
            os.path.join(self.data_dir, 'training'), delim_whitespace=True)
        
        tr_loss = np.mean(df[df.Epoch == epoch].Loss)
        ob_loss = np.mean(df[df.Epoch == epoch].LossObjectness)
        cl_loss = np.mean(df[df.Epoch == epoch].LossClassifier)

        df = pd.read_csv(
            os.path.join(self.data_dir, 'validation'), delim_whitespace=True)

        epoch_ap = df[df.Epoch == epoch]

        prec_values = epoch_ap[epoch_ap.Title == 'AveragePrecision'].Value
        rec_values = epoch_ap[epoch_ap.Title == 'AverageRecall'].Value
        
        ap_p = np.mean(prec_values[prec_values > 0])
        ap_r = np.mean(rec_values[rec_values > 0])
        
        
        self.curr_overall_score = ((2 - tr_loss) / 3) + \
            ((2 - ob_loss) / 3) + ((2 - cl_loss) / 3) + ap_p + ap_r

    def evaluate_model(self) -> bool:
        if self.curr_overall_score > self.last_chkpt_overall_score:
            self.epoch_chkpt = self.epoch
            self.last_chkpt_overall_score = self.curr_overall_score
            return True
        else:
            return False
