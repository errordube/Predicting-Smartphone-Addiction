import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

t_start = time.time()

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
test = pd.read_csv('/mnt/user-data/uploads/test.csv')
target='addicted_label'
cat_cols=['gender','stress_level','academic_work_impact']
num_cols=[c for c in train.columns if c not in cat_cols+['id',target]]

for c in cat_cols:
    train[c]=train[c].astype('category')
    test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

features=num_cols+cat_cols
X=train[features]; y=train[target]
X_test = test[features]

X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)

params = dict(objective='binary',metric='auc',learning_rate=0.05,num_leaves=127,
              min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
              n_estimators=3000,n_jobs=-1,verbosity=-1)

model = lgb.LGBMClassifier(**params, random_state=42)
model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],eval_metric='auc',
          callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])
best_iter = model.best_iteration_
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
print(f"Split model: best_iter={best_iter}, val_auc={val_auc:.5f}, time={time.time()-t_start:.1f}s")

# Retrain on full data using ~best_iter rounds (increase n_seeds below for a
# bagged/averaged ensemble if you have more time budget - each seed adds ~100-200s)
n_final = int(best_iter * 1.05)
test_preds = np.zeros(len(test))
n_seeds = 1
for seed in range(n_seeds):
    m = lgb.LGBMClassifier(**{**params, 'n_estimators': n_final}, random_state=seed)
    m.fit(X, y)
    test_preds += m.predict_proba(X_test)[:,1] / n_seeds
    print(f"seed {seed} done, time={time.time()-t_start:.1f}s")

sub = pd.DataFrame({'id': test['id'], 'addicted_label': test_preds})
sub.to_csv('/mnt/user-data/outputs/submission.csv', index=False)
print("Saved. total time", time.time()-t_start)
