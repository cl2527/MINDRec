import os

import matplotlib.pyplot as plt
import numpy as np
import json
tgt_folder = './plots'
if not os.path.exists(tgt_folder):
    os.makedirs(tgt_folder)

result_root_dir = './results/MIND_30cap/tra_NU10_val_NU1_te_NU1_histLen_30/re_42_2048_1e-4'

#epoch_0_state_json = './results/MIND_30cap/tra_NU10_val_NU1_te_NU1_histLen_30/re_42_2048_1e-4/all_results.json'
#log_history_jason = './results/MIND_30cap/tra_NU10_val_NU1_te_NU1_histLen_30/re_42_2048_1e-4/checkpoint-76/trainer_state.json'
epoch_0_state_json = os.path.join(result_root_dir, 'all_results.json')
log_history_jason = os.path.join(result_root_dir, 'checkpoint-76', 'trainer_state.json')
epoch = []
auc = []
mrr = []
ndcg_5 = []
recall_5 = []
with open(epoch_0_state_json, 'r') as f:
    D_epoch_0 = json.load(f)
    auc.append(D_epoch_0['eval_auc'])
    mrr.append(D_epoch_0['eval_mrr'])
    ndcg_5.append(D_epoch_0['eval_ndcg@5'])
    recall_5.append(D_epoch_0['eval_recall@5'])
    epoch.append(0)

with open(log_history_jason, 'r') as f:
    D = json.load(f)
    log_history = D['log_history']
    # print(log_hisroty)
    
    for log in log_history:
        if 'eval_auc' not in log:
            continue
        auc.append(log['eval_auc'])
        mrr.append(log['eval_mrr'])
        ndcg_5.append(log['eval_ndcg@5'])
        recall_5.append(log['eval_recall@5'])
        epoch.append(log['epoch'])
        
        
improve_auc = (max(auc) - auc[0]) / auc[0] * 100
improve_mrr = (max(mrr) - mrr[0]) / mrr[0] * 100
improve_ndcg_5 = (max(ndcg_5) - ndcg_5[0]) / ndcg_5[0] * 100
improve_recall_5 = (max(recall_5) - recall_5[0]) / recall_5[0] * 100
print(f'AUC improved by {improve_auc:.2f}%, from {auc[0]:.4f} to {max(auc):.4f}')
print(f'MRR improved by {improve_mrr:.2f}%, from {mrr[0]:.4f} to {max(mrr):.4f}')
print(f'NDCG@5 improved by {improve_ndcg_5:.2f}%, from {ndcg_5[0]:.4f} to {max(ndcg_5):.4f}')
print(f'Recall@5 improved by {improve_recall_5:.2f}%, from {recall_5[0]:.4f} to {max(recall_5):.4f}')

plt.figure()
plt.plot(epoch, auc, label='AUC')
plt.plot(epoch, mrr, label='MRR')
plt.plot(epoch, ndcg_5, label='NDCG@5')
plt.plot(epoch, recall_5, label='Recall@5')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Evaluation Metrics ')
plt.legend()
plt.grid()
plt.savefig(os.path.join(tgt_folder, 'metrics.png'))
plt.close()

