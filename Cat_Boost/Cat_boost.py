#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import random

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

import tools
from tools.data import read_data
from tools.metrics import save_divide


# In[ ]:


# configs
filename = 'BTCUSDT-trades-2022-11-25'
RANDOM_STATE = 42

y_cols = ['Y', 'Y2']
#ignored_cols = set(['S_IDNO', 'PHONE', '休眠日', '複委託靜止狀態', '第幾波回來'] + y_cols)


# In[ ]:


# viz config
titlesize = 20
fontsize = 15
labelsize = 15
alpha = 0.8

plt.rcParams['font.family'] = 'Droid Sans Fallback'
plt.rcParams['axes.unicode_minus'] = False


# In[ ]:


def compute_class_weights(y, labels=[]):
    if len(labels) == 0:
        labels = np.unique(y)
    labels = sorted(labels)
    n_classes = np.array([sum(y == label) for label in labels])
    class_weights = 1 / (n_classes) * (1 / sum(1 / n_classes))
    #print(n_classes)
    assert len(y) == sum(n_classes)
    return class_weights


def get_recall_precision(y_pred, y_true, topk):
    n = len(y_pred)
    pred_cols, true_cols = y_pred.columns.tolist(), y_true.columns.tolist()
    result = pd.concat((y_pred, y_true), axis=1)
    tp = result.sort_values(pred_cols, ascending=False).iloc[:topk][true_cols].sum().values[0]
    y_true_sum = y_true.sum().values[0]
    recall = save_divide(tp, y_true_sum)
    precision = tp / topk
    print(f'recall: {recall: .4f}, precision: {precision: .4f}, tp: {tp}, y_true: {y_true_sum}, n: {n}')


def run_k_fold(df, x_cols, y_col, cate_cols, params, topk, k=5, once=False):
    y_col = [y_col]
    
    kf = StratifiedKFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
    result = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df['第幾波回來'])):
        df_train, df_test = df.loc[train_idx], df.loc[test_idx]

        # train
        model = CatBoostClassifier(**params)
        model.fit(df_train[x_cols], df_train[y_col], cat_features=cat_cols, verbose=False)

        # evaluate
        predict = pd.DataFrame(model.predict_proba(df_test[x_cols])[:, 1], index=df_test.index, columns=['predict'])
        get_recall_precision(predict, df_test[y_col], topk)
        result[fold] = {
            'y_pred': predict.predict.to_numpy(),
            'y_true': df_test[y_col[0]].to_numpy(),
            'y_wave': df_test['第幾波回來'].to_numpy(),
            'y_channel': df_test['Y2'].to_numpy(),
        }
        if once:
            break
    return model, df_test, result


# In[ ]:


# read data
y_col = y_cols[0]
df, x_cols, num_cols, cat_cols = read_data(filename, ignored_cols)

# model parameters
topk = 3000
params = {
    'iterations': 1000,
    'depth': 6,
    #'class_weights': compute_class_weights(df[y_col])
}
print(params)
# K fold
model, df_test, result = run_k_fold(df, x_cols, y_col, cat_cols, params, topk, once=True)
#with open('./tmp/result_all_customer.pkl', 'wb') as f:
#   pickle.dump(result, f)


# ## Precision recall curve

# In[ ]:


def draw_precision_recall(df):
    fig, ax1 = plt.subplots(sharex=True,sharey=True,figsize=(16, 7))

    # two y axis
    ax2 = ax1.twinx()

    # plot
    l1 = ax1.plot(df['topk'], df['recall'], 'o-b', label='recall', alpha=alpha)
    l2 = ax2.plot(df['topk'], df['precision'], 'o-g', label='precision', alpha=alpha)

    plot_recall = df.loc[df.plot_recall == True]
    for row in plot_recall.itertuples():
        plt.axvline(x=row.topk,linewidth=1, color='r', ls='--')
        plt.text(row.topk, 0.3, f' recall={row.recall:.1f}%\n topk={row.topk}\n cost={row.proportion:.1f}%')
        
    #l4 = plt.axhline(y=5.5,linewidth=1, color='b')

    # label
    ax1.set_ylabel('recall(%)', fontsize=fontsize)
    ax2.set_ylabel('precision(%)', fontsize=fontsize)
    ax1.set_xlabel('人數', fontsize=fontsize)

    # ticks
    ax1.tick_params(labelsize=labelsize)
    ax2.tick_params(labelsize=labelsize)

    # legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='center right', fontsize=fontsize)

    # title
    plt.title('Precision & Recall & 廣發結果(水平實線)', fontsize=titlesize)

    plt.show()
    

def get_recall_precision_from_kfold(result: dict, n=3000, step=200):
    result = {
        fold: pd.DataFrame(res).sort_values('y_pred', ascending=False).reset_index(drop=True)\
        for fold, res in result.items()
    }
    recall_precision = []
    folds = len(result)
    n_all = sum([len(v) for v in result.values()])
    for k in range(step, n, step):
        tp, y_sum = 0, 0
        for fold, v in result.items():
            tp += result[fold].iloc[:k//folds].y_true.sum()
            y_sum += result[fold].y_true.sum()
        recall_precision.append({
            'topk': k ,
            'recall': save_divide(tp, y_sum) * 100,
            'precision': save_divide(tp, (folds * k)) * 100
        })
    recall_precision = pd.DataFrame(recall_precision)
    recall_precision['proportion'] = recall_precision.topk / n_all * 100
    recall_precision['plot_recall'] = np.zeros(len(recall_precision), dtype=bool)
    for recall in [50, 70, 90]:
        idx = recall_precision.loc[(recall <= recall_precision.recall)].index
        if len(idx) > 0:
            recall_precision.loc[idx[0], 'plot_recall'] = True
    return recall_precision


recall_precision = get_recall_precision_from_kfold(result, n=4000, step=200)
recall_precision

draw_precision_recall(recall_precision)


# ## Analysis - promotion policy
# 
# 1. seting sms_ratio
# 2. sorting the probabilities from model incresingly
# 3. promote every `pr_step` and send `sms_ratio`% sms from low probability to high

# In[ ]:


import math
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

from tools.metrics import cost_performance, plot_cps, plot_sms_pr
from tools.promotion import pr_promote

with open('./tmp/result_all_customer.pkl', 'rb') as f:
    result = pickle.load(f)
print(result)

pd.options.display.float_format = '{:,.3f}'.format

font = {'size'   : 25}
matplotlib.rc('font', **font)

    
def analysis_policies(result, sms_ratio, mode, pr_step=5, verbose=False):
    analysis_res = pd.DataFrame(columns=['sms_pr', 'tp', 'cost', 'tp_std',  'cost_std'])

    res = fold_pr_evaluate(result, sms_ratio, mode, None, n_times=10)
    analysis_res.loc[len(analysis_res)] = ['random', np.mean(res['tps']), np.mean(res['costs']),                                           np.std(res['tps']), np.std(res['costs'])]
    sms_amount = int(sms_ratio * 100)
    for pr in range(0, 100, pr_step):
        i = len(analysis_res)
        if sms_amount + pr > 101:
            continue
        res = fold_pr_evaluate(result, sms_ratio, mode, pr)
        analysis_res.loc[i] = [pr, np.mean(res['tps']), np.mean(res['costs']),                               np.std(res['tps']), np.std(res['costs'])]
    analysis_res['cp'] = analysis_res.tp / analysis_res.cost
    analysis_res = analysis_res[::-1].reset_index(drop=True)
    if verbose:
        print(analysis_res)
    return analysis_res

def analysis_different_sms_ratio(result, start, end, step, mode):
    analysis_res = pd.DataFrame(columns=['SMS比率', '喚醒', '成本'])
    sms_ratio = start
    while sms_ratio < end:
        pr = 100 - sms_ratio * 100
        res = fold_pr_evaluate(result, sms_ratio, mode, pr)
        
        i = len(analysis_res)
        analysis_res.loc[i] = [sms_ratio, np.mean(res['tps']), np.mean(res['costs'])]
        
        sms_ratio += step
    analysis_res['CP值'] = analysis_res.iloc[:, 1] / analysis_res.iloc[:, 2]
    return analysis_res

def fold_pr_evaluate(result, sms_ratio, mode, pr_start=None, n_times=1):
    res_eval = {'tps': [], 'costs': [], 'n_times': n_times}
    for _ in range(n_times):
        tps, costs = [], []
        for fold, res in result.items():
            sms_idx = pr_promote(res['y_pred'], sms_ratio, pr_start)

            y = pd.DataFrame({'channel': res['y_channel'], 'wave': res['y_wave']})
            y_sms = y.iloc[sms_idx].copy()
            y_edm = y.drop(sms_idx).copy()
            tp, cost = cost_performance(y_sms, y_edm, mode)
            tps.append(tp)
            costs.append(cost)
        res_eval['tps'].append(np.sum(tps))
        res_eval['costs'].append(np.sum(costs))
    return res_eval

# fix sms_ratio
sms_ratio = 0.18

modes = ['strict', 'SMS>EDM', 'loose']
mode_mapping = {
    'strict': '嚴格',
    'SMS>EDM': '寬鬆(SMS較強)',
    'loose': '寬鬆',
}
if True:
    results = []
    for mode in modes:
        res = analysis_policies(result, sms_ratio, mode, pr_step=1, verbose=True)
        results.append(res)
    plot_cps(results, plot_sms_pr, title=f'SMS比率_{sms_ratio}_機率分段分析', subtitles=[mode_mapping[m] for m in modes],
             y_min1=0, y_max1=500, y_min2=0, y_max2=16000)
    plt.show()
if True:
    results = []
    for mode in modes:
        df = analysis_different_sms_ratio(result, 0., 0.18, 0.01, mode)
        results.append(df)
        print(df)
    plot_cps(results, title=f'SMS比率分析', subtitles=[mode_mapping[m] for m in modes],
             y_mins=[0, 0, 0], y_maxs=[500, 16000, 0.1])
        
''


# ## make promotion list

# In[ ]:


import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from tools.data import read_data
from tools.promotion import pr_promote


def predict(model, df, x_cols, cat_cols):
    proba = model.predict_proba(df[x_cols])[:, 1]
    idx = np.argsort(proba)
    res = pd.DataFrame({
        'S_IDNO': df.loc[idx].S_IDNO,
        'proba': proba[idx],
    })[::-1].reset_index(drop=True)
    return res


y_cols = ['Y', 'Y2']
ignored_cols = set(['S_IDNO', 'PHONE', '休眠日', '複委託靜止狀態', '第幾波回來'] + y_cols)

y_col = y_cols[0]
df, x_cols, num_cols, cat_cols = read_data('./data/CUSTOMER_202207__wave_toy.xlsx', ignored_cols)
df_te, x_cols_te, num_cols_te, cat_cols_te = read_data('./data/CUSTOMER_202207__wave_toy_te.xlsx', ignored_cols)
    
# model parameters
params = {
    'iterations': 1000,
    'depth': 6,
    #'class_weights': compute_class_weights(df[y_col])
}

# train
model = CatBoostClassifier(**params)
model.fit(df[x_cols], df[y_col], cat_features=cat_cols, verbose=False)
res = predict(model, df_te, x_cols_te, cat_cols_te)
res


# In[ ]:


sms_ratio = 0.18


pr_start = 100 - sms_ratio * 100
sms_idx = pr_promote(res.proba, sms_ratio, pr_start)
edm_idx = [i for i in range(len(res)) if i not in sms_idx]
res['promotion'] = None
res.loc[sms_idx, 'promotion'] = 'SMS'
res.loc[edm_idx, 'promotion'] = 'EDM'
print(res)
res.to_csv('./tmp/model_1_promotions.csv')

