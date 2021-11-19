
import os
import numpy as np
import pandas as pd


class EliteModel():
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        self.curr_overall_score = 0
        self.last_chkpt_overall_score = -1
        
    def calculate_metrics(self, epoch: int):
        df = pd.read_csv(
            os.path.join(self.data_dir, 'validation.txt'), delim_whitespace=True)

        epoch_ap = df[df.Epoch == epoch]

        prec_values = epoch_ap[epoch_ap.Title == 'AveragePrecision'].Value
        rec_values = epoch_ap[epoch_ap.Title == 'AverageRecall'].Value
        
        ap_p = np.mean(prec_values[prec_values >= 0])
        ap_r = np.mean(rec_values[rec_values >= 0])
        
        self.curr_overall_score = ap_p + ap_r

    def evaluate_model(self) -> bool:
        if self.curr_overall_score > self.last_chkpt_overall_score:
            self.last_chkpt_overall_score = self.curr_overall_score
            return True
        else:
            return False
