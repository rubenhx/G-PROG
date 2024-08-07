import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

import model as M
import util as U
from dataGRAPE import GlauProg_Dataset

EXPERIMENT = '00000_ENB0relu_smoothL1_gt5_RL'


global params
params = {
    'device': 'cuda',
    'project_dir': '../experiments/grisk_qle-2_es1_sh1_dd1_hu0_g0d0_g1d0_imgc0_ymin2_ymax5_vis3_mdiff3000/baseline_grisk60/',
    'experiment': EXPERIMENT,
    'model': EXPERIMENT.split('_')[1],
    'loss': EXPERIMENT.split('_')[2],
    'batchsize': 1,
    'img_h': 512,
    'img_w': 512,
    'auc_th': 2,
    'weightname': 'explvar_weights.pt',
    'saveresults': True,
    }

val_transform = A.Compose(
    [
        A.Resize(params['img_h'], params['img_w'])
    ]
)

def validate(val_loader, model, criterion, params):
    preds = []
    targets = []

    metric_monitor = U.MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream):
            images = images.permute(0,3,1,2)/255.
            images = images.to(params['device'], non_blocking=True).float()
            target = target.to(params['device'], non_blocking=True).float().view(-1, 1)
            output = model(images).float().view(-1, 1)
            loss = criterion(output, target)
            
            metric_monitor.update('Loss', loss.item())
            stream.set_description(
                'Validation. {metric_monitor}'.format(metric_monitor=metric_monitor)
            )

            preds.append(np.squeeze(output.cpu().detach().numpy()).tolist())
            targets.append(np.squeeze(target.cpu().detach().numpy()).tolist())
    val_output = {'valloss': metric_monitor.get_avgloss('Loss'), 'targets': targets, 'preds': preds}
    return val_output

def main():

    device = params['device']
    print(f'Using {device} device')

    print('Inference for experiment {} on GRAPE data ...'.format(params['experiment']))

    val_dataset = GlauProg_Dataset(imglab_csv="GRAPE\proglabels\labels_qle-2_es0_sh0_dd1_hu0_ymin2_ymax5_vis3_mdiff2000_withmd.csv", transform=val_transform)
    dataloader_val = DataLoader(val_dataset, batch_size=params['batchsize'], shuffle=False, num_workers=0)

    # Load model here
    print('Loading model and trained weights ...')
    if params['model'] == 'ENB0':
        model_ft = M.get_ENB0()
    elif params['model'] == 'ENB0relu':
        model_ft = M.get_ENB0relu()
    elif params['model'] == 'RN50relu':
        model_ft = M.get_RN50relu()
    ckp = torch.load(os.path.join(params['project_dir'], params['experiment'], params['weightname']))
    model_ft.load_state_dict(ckp['state_dict'])
    model_ft = model_ft.to('cuda')

    if params['loss'] == 'smoothL1':
        criterion = nn.SmoothL1Loss()

    val_output = validate(dataloader_val, model_ft, criterion, params)

    results = U.evaluate_imgs(np.array(val_output['targets']), np.array(val_output['preds']))
    #auc = roc_auc_score(np.array(val_output['targets'])>=params['auc_th'], np.array(val_output['preds']))
    #print('The AUC is {} at a threshold of {}.'.format(np.round(auc, 3), params['auc_th']))
    prevprog = np.round(np.sum(np.array(val_output['targets'])>=params['auc_th'])/val_dataset.__len__(), 3)
    print('The proportion of progressors is equal to {}.'.format(prevprog))

    print('This inference run obtained \nR2 score of {}, \nPearson r of {}, \nMAE of {}, \nExplained variance of {}'.format(np.round(results['R2'], 3), np.round(results['r'], 3), np.round(results['MAE'], 3), np.round(results['varexpl'], 3)))
    print('The baseline MAE was decreased by {} %'.format(results['mae_decr']*100))

    if params['saveresults'] == True:
        df = pd.read_csv("GRAPE\proglabels\labels_qle-2_es0_sh0_dd1_hu0_ymin2_ymax5_vis3_mdiff2000_withmd.csv")
        df['targets'] = val_output['targets']
        df['preds'] = val_output['preds']
        print('{}{}/output_GRAPE.xlsx'.format(params['project_dir'], params['experiment']))
        df.to_excel('{}{}/output_GRAPE.xlsx'.format(params['project_dir'], params['experiment']), index=False)
         
if __name__ == '__main__':
    main()

