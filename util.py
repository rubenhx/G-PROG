from collections import defaultdict
import re
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score

def _get_next_run_id_local(run_dir_root: str) -> int:
    '''Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names.'''
    dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    r = re.compile('^\\d+')  # match one or more digits at the start of the string
    run_id = 0
    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)
    return run_id

def _create_experiment_dir(project_dir, proplist):
    experiment_name = '{0:05d}_{1}'.format(_get_next_run_id_local(project_dir), proplist)
    weight_dir = project_dir + experiment_name + '/'
    print('Created results dir at: ' + str(weight_dir) + ' ...')
    if os.path.exists(weight_dir):
        pass
    else:
        os.mkdir(weight_dir)
    return experiment_name

def get_baselineMAE(labels):
    return mean_absolute_error(labels, np.tile(np.mean(labels, axis=0), (labels.shape[0], 1)))

def evaluate_imgs(labels, preds):
    r2 = r2_score(labels, preds)
    varexpl = explained_variance_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r = np.corrcoef(labels.ravel(), preds.ravel())[0][1]

    _meanlab = np.mean(labels)
    dummypreds = np.tile(_meanlab, reps = len(labels))
    basemae = mean_absolute_error(labels, dummypreds)
    mae_decr = (basemae - mae)/basemae
    return {'R2': r2, 'MAE': mae, 'r': r, 'varexpl': varexpl, 'mae_decr': mae_decr}

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'val': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return ' | '.join(
            [
                '{metric_name}: {avg:.{float_precision}f}'.format(
                    metric_name=metric_name, avg=metric['avg'], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

    def get_avgloss(self, metric_name):
        metric = self.metrics[metric_name]
        return metric['avg']


