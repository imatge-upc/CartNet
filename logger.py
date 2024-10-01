import logging
import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.metrics import r2_score
from torch_geometric.graphgym import get_current_gpu_usage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.logger import infer_task, Logger
from torch_geometric.graphgym.utils.io import dict_to_json
from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class CustomLogger(Logger):
    def __init__(self,*args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self._size_current_ellipse = 0
    #     # Whether to run comparison tests of alternative score implementations.
    #     self.test_scores = False

    def reset(self):
        self._iter = 0
        self._size_current = 0
        self._size_current_ellipse = 0
        self._loss = 0
        self._lr = 0
        self._params = 0
        self._time_used = 0
        self._true = []
        self._pred = []
        self._class = []
        self._custom_stats = {}

    # basic properties
    def basic(self):
        stats = {
            'loss': self._loss / self._size_current,
            'lr': self._lr,
            'params': self._params,
            'time_iter': round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats


    def mae_mape_per_class(self, pred, true, classes):
        """
        Compute the Mean Absolute Error (MAE) and Mean Percentage Absolute Error (MAPE) per class and then average between classes.

        Args:
        pred (torch.Tensor): Predicted values tensor.
        true (torch.Tensor): Ground truth values tensor.
        classes (torch.Tensor): Class labels tensor.

        Returns:
        float: Averaged MAE between classes.
        """
        # Calculate the absolute errors
        abs_errors = torch.abs(pred - true)

        #Calculate percentage errors
        percent_errors = abs_errors/(true+1e-8)

        # Get unique classes and their counts
        unique_classes, counts = classes.unique(return_counts=True)

        # Initialize a tensor to accumulate the sum of absolute errors per class
        abs_error_sums = torch.zeros_like(unique_classes, dtype=torch.float)

        # Initialize a tensor to accumulate the sum of percentage errors per class
        percent_error_sums = torch.zeros_like(unique_classes, dtype=torch.float)

        # Accumulate the absolute errors per class using scatter_add
        abs_error_sums.scatter_add_(0, classes, abs_errors)

        # Accumulate the percentage errors per class using scatter_add
        percent_error_sums.scatter_add_(0, classes, percent_errors)

        # Calculate the mean absolute error per class
        mae_per_class = abs_error_sums / counts.float()

        # Calculate the mean absolute percentage error per class
        mape_per_class = percent_error_sums / counts.float()

        # Calculate the average MAE across classes
        avg_mae = mae_per_class.mean()

        # Calculate the average MAPE across classes
        avg_mape = mape_per_class.mean()

        return avg_mae.item(), avg_mape.item()

    def regression(self):
        batch_size = 1e6
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: float(x)
        if len(self._class):
            classes = torch.cat(self._class)
            avg_mae_per_class, avg_mape_per_class = self.mae_mape_per_class(pred, true, classes)
            return {
                'r2': reformat(eval_r2(true.numpy(), pred.numpy())),
                'spearmanr': reformat(eval_spearmanr(true.numpy(),
                                                    pred.numpy())['spearmanr']),
                "MAE_per_class": avg_mae_per_class,
                "MAPE_per_class": avg_mape_per_class,
            
            }

            

        print(true.dtype)
        print("True")
        print(true[0])
        print("Pred")
        print(pred[0])
        
        try:
            return {
                'r2': reformat(eval_r2(true.numpy(), pred.numpy())),
                'spearmanr': reformat(eval_spearmanr(true.numpy(),
                                                    pred.numpy())['spearmanr']),
            
            }
        except:
            return {}
        
    def custom(self):
        if len(self._custom_stats) == 0:
            return {}
        out = {}
        for key, val in self._custom_stats.items():
            if key in ["loss_ellipse", "volume_percentage_error"] and self._size_current_ellipse>0:
                out[key] = val / self._size_current_ellipse
            else:
                out[key] = val / self._size_current
        return out


    def update_stats(self, true, pred, loss, lr, time_used, params,
                     dataset_name=None, classes=None, batch_ellipse=None, **kwargs):
        
        assert true.shape == pred.shape, (true.shape, pred.shape)
        batch_size = true.shape[0]
        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        if classes is not None:
            self._class.append(classes)
        self._size_current += batch_size
        self._loss += loss*batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        if batch_ellipse is not None:
            self._size_current_ellipse += batch_ellipse
        for key, val in kwargs.items():
            if batch_ellipse is not None and key in ["loss_ellipse", "volume_percentage_error"]:
                if key not in self._custom_stats:
                    self._custom_stats[key] = val * batch_ellipse
                else:
                    self._custom_stats[key] += val * batch_ellipse
            else:
                if key not in self._custom_stats:
                    self._custom_stats[key] = val * batch_size
                else:
                    self._custom_stats[key] += val * batch_size

    def write_epoch(self, cur_epoch):
        start_time = time.perf_counter()
        basic_stats = self.basic()

        if self.task_type == 'regression':
            task_stats = self.regression()
        elif self.task_type == 'classification_binary':
            task_stats = self.classification_binary()
        elif self.task_type == 'classification_multi':
            task_stats = self.classification_multi()
        elif self.task_type == 'classification_multilabel':
            task_stats = self.classification_multilabel()
        elif self.task_type == 'subtoken_prediction':
            task_stats = self.subtoken_prediction()
        else:
            raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch,
                       'time_epoch': round(self._time_used, cfg.round)}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round),
                     'eta_hours': round(self.eta(cur_epoch) / 3600, cfg.round)}
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {
                **epoch_stats,
                **eta_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }
        else:
            stats = {
                **epoch_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        
        self.reset()
        if cur_epoch < 3:
            logging.info(f"...computing epoch stats took: "
                         f"{time.perf_counter() - start_time:.2f}s")
        return stats


def create_logger():
    """
    Create logger for the experiment

    Returns: List of logger objects

    """
    loggers = []
    names = ['train', 'val', 'test']
    for name in names:
        loggers.append(CustomLogger(name=name, task_type="regression"))
    return loggers
    


def eval_spearmanr(y_true, y_pred):
    """Compute Spearman Rho averaged across tasks.
    """
    res_list = []

    if y_true.ndim == 1:
        res_list.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(stats.spearmanr(y_true[is_labeled, i],
                                            y_pred[is_labeled, i])[0])

    return {'spearmanr': sum(res_list) / len(res_list)}

def eval_r2(y_true, y_pred):
    """Compute Spearman Rho averaged across tasks.
    """
    res_list = []
    print(y_true.shape)
    print(y_pred.shape)

    if y_true.ndim == 1:
        res_list.append(r2_score(y_true, y_pred, multioutput='uniform_average'))
    else:
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(r2_score(y_true[is_labeled, i],
                                            y_pred[is_labeled, i], multioutput='uniform_average'))

    return sum(res_list) / len(res_list)
