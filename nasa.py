
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.metrics import accuracy_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import plot_importance
from lightgbm import plot_metric
import warnings
from sklearn.metrics.classification import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import bisect
import numpy as np
from pylab import mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import smote_variants as sv

#加载数据
df = pd.read_csv('./KC1.arff.csv')
#查看表的基本特征，展开省略号
pd.set_option('display.max_columns',1000)
print(df)
#查看缺失值
print(df.isnull().sum())
#查看大小
print(df.shape)
# 查看样本以及样本缺陷个数,df.Defective做一下筛选
print('样本个数：{}'.format(df.shape[0]))
print('无缺陷个数：{}'.format(df[df.Defective=='N'].shape[0]))
print('有缺陷个数：{}'.format(df[df.Defective=='Y'].shape[0]))
#分离特征值与label
#对X  [:,:-1]不要最后一列  ,对y [:,-1]要最后一列
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#使用标签编码
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
print(y)
#进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
#切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#使用SVC进行预测
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('SVM 预测结果：',y_pred)
print('SVM 预测准确率：',accuracy_score(y_test,y_pred))

#换种方式求准确率
print('SVM 预测准确率：',svc.score(X_test,y_test))
#SVC中换线性核预测
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('linear SVM 预测结果：',y_pred)
print('linear SVM 预测准确率：',accuracy_score(y_test,y_pred))
#使用RF进行预测
rf=RandomForestClassifier(random_state=3)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Random Forest 预测结果：',y_pred)
print('Random Forest 预测准确率：',accuracy_score(y_test,y_pred))
#使用KNN进行预测
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_knn = svc.predict(X_test)
y_knn_prob=rf.predict_proba(X_test)[:,1]
Recall=metrics.recall_score(y_test,y_knn)
Auc=metrics.roc_auc_score(y_test,y_knn)
print("KNN_Recall:\n",metrics.recall_score(y_test,y_knn))
#print("G_mean:\n",gmean)
print("KNN_AUC:\n",metrics.roc_auc_score(y_test,y_knn_prob))
print('KNN 预测结果：',y_pred)
print('KNN 预测准确率：',accuracy_score(y_test,y_knn))
#使用LR进行预测
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('LR 预测结果：',y_pred)
print('LR 预测准确率：',accuracy_score(y_test,y_pred))

#使用xgboost进行预测
param = {'boosting_type':'gbdt',
                         'objective' : 'binary:logistic', #任务目标
                         'eval_metric' : 'auc', #评估指标
                         'eta' : 0.01, #学习率
                         'max_depth' : 15, #树最大深度
                         'colsample_bytree':0.8, #设置在每次迭代中使用特征的比例
                         'subsample': 0.9, #样本采样比例
                         'subsample_freq': 8, #bagging的次数
                         'alpha': 0.6, #L1正则
                         'lambda': 0, #L2正则
}
#XGBoost无法解析带有标头的CSV文件，使用Xgboost自带的读取格式DMatrix()
train_data = xgb.DMatrix(X_train, y_train)
test_data = xgb.DMatrix(X_test, y_test)
#早停法，设2000次，如果37次结果没变化就停止
model = xgb.train(param, train_data,evals=[(train_data,'train'),(test_data,'valid')],num_boost_round=4000,early_stopping_rounds=37,verbose_eval=37)
y_pred = model.predict(test_data)
#将预测值做0，1处理
y_pred = [1 if x>=0.5 else 0 for x in y_pred]
print('xgboost 预测结果:',y_pred)
print('xgboost 预测准确率：',accuracy_score(y_test,y_pred))
#使用lightgbm进行训练
is_PolynomialFeatures = True
is_predict = True
ia_GridSearchCV = False
algorithm = 'lightgbm'
def model_processing(X, y, category_list):
    oversampler = sv.AND_SMOTE(proportion=2, random_state=3)
    X_columns = X.columns.tolist()
    y_columns = y.columns.tolist()
    X_array = np.array(X)
    y_array = np.array(y).reshape(1, -1)[0]
    X_samp, y_samp = oversampler.sample(X_array, y_array)

    X_raw = X.copy()
    y_raw = y.copy()
    print(X_raw.info())
    X = pd.DataFrame(X_samp, columns=X_columns)
    y = pd.DataFrame(y_samp, columns=y_columns)
    for col_name in X.columns.tolist():
        X[col_name] = X[col_name].map(float)
    if is_predict:
        train_X = X
        train_y = y
    else:
        train_X, test_X, train_y, test_y = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=15)
    if algorithm == 'lightgbm':
        # 是否网格搜索
        if ia_GridSearchCV:
            param = {
                # 'metric': ['binary_logloss', 'auc'],
                # 'num_leaves': [16, 18, 20, 22, 24],
                # 'min_data_in_leaf': [18, 20, 22, 24, 26, 28, 30],
                # 'max_steps': [3, 4, 5],
                # 'learning_rate': [0.03, 0.06, 0.1, 0.13]
            }
            model = GridSearchCV(
                estimator=LGBMClassifier(
                    task='train',
                    boosting_type='gbdt',
                    objective='binary',
                    metric='auc',
                    num_leaves=20,
                    min_data_in_leaf=24,
                    n_estimators=160,
                    learning_rate=0.1,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    max_steps=3,
                    #   verbose=2,
                    num_class=1),
                param_grid=param,
                scoring='roc_auc',
                cv=2,
                verbose=2)
            fit_params = {'verbose': False}
            model.fit(X, y, **fit_params)
            print(model.best_params_, model.best_score_)
            return

        train_params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'binary',  # 目标函数
            'metric': 'auc',  # 评估函数
            'num_leaves': 18,  # 叶子节点数
            'min_data_in_leaf': 20,  # 每个叶子最小叶子数
            'max_steps': 3,
            'learning_rate': 0.1,  # 学习速率
            'feature_fraction': 0.8,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 2,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
            'n_estimators': 130,
            'num_class': 1,  # label的个数
            'seed': 99
        }
        #model = lightgbm.train(param, train_data,evals=[(train_data,'train'),(test_data,'valid')],num_boost_round=4000,early_stopping_rounds=37,verbose_eval=37)
        if is_predict:
            if encoder == 'LabelEncoder':
                fit_params = {
                    'verbose': False,
                    # 'categorical_feature': list(category_list),
                    # 'eval_set': [(train_X, train_y), (test_X, test_y)],
                    # 'eval_metric': ['binary_logloss'],  # 'binary_logloss'
                    # 'early_stopping_rounds': 50
                }
            else:
                fit_params = {
                    'verbose': False,
                    # 'eval_set': [(train_X, train_y), (test_X, test_y)],
                    # 'categorical_feature': list(category_list),
                    # 'eval_metric': ['binary_logloss'],  # 'binary_logloss'
                    # 'early_stopping_rounds': 50
                }
        else:
            if encoder == 'LabelEncoder':
                fit_params = {
                    'verbose': False,
                    # 'categorical_feature': list(category_list),
                    'eval_set': [(train_X, train_y), (test_X, test_y)],
                    'eval_metric': ['auc'],  # 'binary_logloss'
                    'early_stopping_rounds': 30
                }
            else:
                fit_params = {
                    'verbose': False,
                    'eval_set': [(train_X, train_y), (test_X, test_y)],
                    'eval_metric': ['auc'],  # 'binary_logloss'
                    'early_stopping_rounds': 30
                }
        model = LGBMClassifier(**train_params)
        print(train_X.info())
        output_model = model.fit(train_X, train_y, **fit_params)
        if is_predict:
            return output_model
        else:
            train_X = X_raw
            train_y = y_raw
            test_X = X_raw
            test_y = y_raw
            train_X_pred = output_model.predict(train_X)
            test_X_pred = output_model.predict(test_X)
            y_pred_pro = output_model.predict_proba(test_X)
            y_scores = pd.DataFrame(
                y_pred_pro, columns=output_model.classes_.tolist())[1].values
            # 输出重要性
            plot_importance(output_model,
                            importance_type='gain',
                            max_num_features=15,
                            ignore_zero=True,
                            figsize=(25, 10),
                            precision=3)
            plt.show()
            plt.close()
            # 输出最好的分数
            print('best_score_ is {}'.format(output_model.best_score_))
            # 输出最佳迭代次数
            best_iteration = output_model.best_iteration_
            print('best_iteration_ is {}'.format(best_iteration))
            # 输出metric图
            _, ax = plt.subplots(figsize=(20, 10))
            plot_metric(output_model, ax=ax, metric='auc')
            plt.savefig('./CM1.png')
            plt.close()
            # 输出训练集上的混淆矩阵
            print('train_confusion_matrix is: ')
            train_confusion = confusion_matrix(train_y,
                                               train_X_pred,
                                               labels=[0, 1])
            print(train_confusion)
            plt.imshow(train_confusion, cmap=plt.cm.Blues)
            indices = range(len(train_confusion))
            classes = ['N', 'Y']
            plt.xticks(indices, classes)
            plt.yticks(indices, classes)
            plt.colorbar()
            plt.xlabel('预测值')
            plt.ylabel('真实值')
            plt.title('train_confusion_matrix')

            for first_index in range(len(train_confusion)):
                for second_index in range(len(train_confusion[first_index])):
                    plt.text(first_index, second_index,
                             train_confusion[first_index][second_index])
            plt.savefig('././CM1_2.png')
            plt.close()
            # 输出测试集集上的混淆矩阵
            print('valid_confusion_matrix is: ')
            test_confusion = confusion_matrix(test_y,
                                              test_X_pred,
                                              labels=[0, 1])
            print(test_confusion)
            plt.imshow(test_confusion, cmap=plt.cm.Blues)
            indices = range(len(test_confusion))
            classes = ['N', 'Y']
            plt.xticks(indices, classes)
            plt.yticks(indices, classes)
            plt.colorbar()
            plt.xlabel('预测值')
            plt.ylabel('真实值')
            plt.title('test_confusion_matrix')

            for first_index in range(len(test_confusion)):
                for second_index in range(len(test_confusion[first_index])):
                    plt.text(first_index, second_index,
                             test_confusion[first_index][second_index])
            plt.savefig('./CM1_3.png')
            plt.close()
            print(classification_report(test_y, test_X_pred))

            # 绘制ROC曲线
            auc_value = roc_auc_score(test_y, y_scores)
            fpr, tpr, thresholds = roc_curve(test_y, y_scores, pos_label=1.0)
            plt.figure()
            lw = 2
            plt.plot(fpr,
                     tpr,
                     color='darkorange',
                     linewidth=lw,
                     label='ROC curve (area = %0.4f)' % auc_value)
            plt.plot([0, 1], [0, 1],
                     color='navy',
                     linewidth=lw,
                     linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.savefig('./CM1_4.png')
            plt.close()
    elif algorithm == 'RF':
        # 是否网格搜索
        if ia_GridSearchCV:
            param = {
                'max_depth': [3, 4, 5, 6],
                'max_features': [0.7, 0.8, 0.9, 1],
                'min_samples_split': [6, 10, 20],
                'min_samples_leaf': [10, 20, 30, 40],
                'warm_start': [True, False],
                'oob_score': [True, False],
                'verbose': [True, False]
            }
            GSCV_model = GridSearchCV(estimator=RandomForestClassifier(),
                                      param_grid=param,
                                      scoring='roc_auc',
                                      cv=2,
                                      verbose=2)
            GSCV_model.fit(X, y)
            print(GSCV_model.best_params_, GSCV_model.best_score_)
            return

        model = RandomForestClassifier(n_estimators=500,
                                       max_features=0.8,
                                       min_samples_split=12,
                                       min_samples_leaf=20,
                                       max_depth=6)
        output_model = model.fit(train_X, train_y)
        if is_predict:
            return output_model
        else:
            y_pred = output_model.predict(test_X)
            y_pred_pro = output_model.predict_proba(test_X)
            y_scores = pd.DataFrame(
                y_pred_pro, columns=output_model.classes_.tolist())[1].values
            print(classification_report(test_y, y_pred))
            auc_value = roc_auc_score(test_y, y_scores)

            # 绘制ROC曲线
            fpr, tpr, thresholds = roc_curve(test_y, y_scores, pos_label=1.0)
            plt.figure()
            lw = 2
            plt.plot(fpr,
                     tpr,
                     color='darkorange',
                     linewidth=lw,
                     label='ROC curve (area = %0.4f)' % auc_value)
            plt.plot([0, 1], [0, 1],
                     color='navy',
                     linewidth=lw,
                     linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.savefig('./CM1_5.png')
            plt.close()
    elif algorithm == 'LR':
        # 是否网格搜索
        if ia_GridSearchCV:
            param = {'max_iter': [500, 1000, 2000]}
            GSCV_model = GridSearchCV(
                estimator=LogisticRegression(class_weight='balanced'),
                param_grid=param,
                scoring='roc_auc',
                cv=2,
                verbose=2)
            GSCV_model.fit(X, y)
            print(GSCV_model.best_params_, GSCV_model.best_score_)
            return

        lr = LogisticRegression(n_jobs=-1,
                                class_weight='balanced',
                                max_iter=300,
                                random_state=50)
        print(train_X.info())
        print(train_y.info())
        output_model = lr.fit(train_X, train_y)
        train_X_pred = output_model.predict(train_X)
        if is_predict:
            return output_model
        else:
            test_X_pred = output_model.predict(test_X)
            y_pred_pro = output_model.predict_proba(test_X)

            y_scores = pd.DataFrame(
                y_pred_pro, columns=output_model.classes_.tolist())[1].values
            print("准确率{}".format(accuracy_score(test_y, test_X_pred)))
            print("分类报告")
            print(classification_report(test_y, test_X_pred))
            print(output_model.coef_)
            print(output_model.intercept_)
            auc_value = roc_auc_score(test_y, y_scores)
            # 绘制ROC曲线
            fpr, tpr, thresholds = roc_curve(test_y, y_scores, pos_label=1.0)
            plt.figure()
            lw = 2
            plt.plot(fpr,
                     tpr,
                     color='darkorange',
                     linewidth=lw,
                     label='ROC curve (area = %0.4f)' % auc_value)
            plt.plot([0, 1], [0, 1],
                     color='navy',
                     linewidth=lw,
                     linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
            print(output_model.coef_)
            plt.savefig('./CM1_6.png')
            plt.close()
    elif algorithm == 'SVC':
        svc_model = make_pipeline(
            StandardScaler(),
            SVC(
                gamma='auto',
                #   kernel='poly',
                decision_function_shape='ovr',
                probability=True))
        output_model = svc_model.fit(train_X, train_y)
        if is_predict:
            return output_model
        else:
            test_X_pred = output_model.predict(test_X)
            y_pred_pro = output_model.predict_proba(test_X)
            y_scores = pd.DataFrame(
                y_pred_pro, columns=output_model.classes_.tolist())[1].values
            print(classification_report(test_y, test_X_pred))
            auc_value = roc_auc_score(test_y, y_scores)

            # 绘制ROC曲线
            fpr, tpr, thresholds = roc_curve(test_y, y_scores, pos_label=1.0)
            plt.figure()
            lw = 2
            plt.plot(fpr,
                     tpr,
                     color='darkorange',
                     linewidth=lw,
                     label='ROC curve (area = %0.4f)' % auc_value)
            plt.plot([0, 1], [0, 1],
                     color='navy',
                     linewidth=lw,
                     linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
            plt.savefig('./CM1_7.png')
            plt.close()
"""     LOC_BLANK  BRANCH_COUNT  CALL_PAIRS  LOC_CODE_AND_COMMENT  LOC_COMMENTS  \
0            6             9           2                     1             0   
1           15             7           3                     1            19   
2           27             9           1                     4            22   
3            7             3           2                     0             0   
4           51            25          13                     0            14   
..         ...           ...         ...                   ...           ...   
339         17            21           4                     1             8   
340         16            13           6                     0             5   
341         11             9           6                     0             4   
342          1             3           0                     0             0   
343         10            13           6                     1            16   

     CONDITION_COUNT  CYCLOMATIC_COMPLEXITY  CYCLOMATIC_DENSITY  \
0                 16                      5                0.20   
1                 12                      4                0.13   
2                 16                      5                0.15   
3                  4                      2                0.17   
4                 48                     13                0.12   
..               ...                    ...                 ...   
339               36                     11                0.26   
340               24                      7                0.12   
341               16                      5                0.12   
342                4                      2                0.22   
343               24                      7                0.18   

     DECISION_COUNT  DECISION_DENSITY  DESIGN_COMPLEXITY  DESIGN_DENSITY  \
0                 8              2.00                  3            0.60   
1                 6              2.00                  2            0.50   
2                 8              2.00                  3            0.60   
3                 2              2.00                  2            1.00   
4                24              2.00                 11            0.85   
..              ...               ...                ...             ...   
339              16              2.25                  5            0.45   
340              12              2.00                  6            0.86   
341               8              2.00                  5            1.00   
342               2              2.00                  1            0.50   
343              12              2.00                  5            0.71   

     EDGE_COUNT  ESSENTIAL_COMPLEXITY  ESSENTIAL_DENSITY  LOC_EXECUTABLE  \
0            17                     1                0.0              24   
1            17                     1                0.0              31   
2            18                     1                0.0              29   
3             6                     1                0.0              12   
4            96                     1                0.0             106   
..          ...                   ...                ...             ...   
339          43                     7                0.6              41   
340          31                     1                0.0              59   
341          24                     1                0.0              42   
342           5                     1                0.0               9   
343          29                     1                0.0              39   

     PARAMETER_COUNT  HALSTEAD_CONTENT  HALSTEAD_DIFFICULTY  HALSTEAD_EFFORT  \
0                  3             32.54                 9.50          2936.77   
1                  1             38.55                21.52         17846.19   
2                  0             52.03                12.33          7914.68   
3                  2             14.62                 6.43           604.36   
4                  2            101.31                27.04         74089.87   
..               ...               ...                  ...              ...   
339                5             42.16                35.93         54431.16   
340                4             95.60                31.40         94278.09   
341                1             66.60                18.54         22884.16   
342                2             14.83                 3.50           181.62   
343                5             29.50                36.25         38771.24   

     HALSTEAD_ERROR_EST  HALSTEAD_LENGTH  HALSTEAD_LEVEL  HALSTEAD_PROG_TIME  \
0                  0.10               63            0.11              163.15   
1                  0.28              141            0.05              991.46   
2                  0.21              111            0.08              439.70   
3                  0.03               23            0.16               33.58   
4                  0.91              421            0.04             4116.10   
..                  ...              ...             ...                 ...   
339                0.50              241            0.03             3023.95   
340                1.00              449            0.03             5237.67   
341                0.41              209            0.05             1271.34   
342                0.02               15            0.29               10.09   
343                0.36              185            0.03             2153.96   

     HALSTEAD_VOLUME  MAINTENANCE_SEVERITY  MODIFIED_CONDITION_COUNT  \
0             309.13                  0.20                         4   
1             829.45                  0.25                         3   
2             641.73                  0.20                         4   
3              94.01                  0.50                         1   
4            2739.78                  0.08                        12   
..               ...                   ...                       ...   
339          1514.78                  0.64                        10   
340          3002.24                  0.14                         6   
341          1234.54                  0.20                         4   
342            51.89                  0.50                         1   
343          1069.55                  0.14                         6   

     MULTIPLE_CONDITION_COUNT  NODE_COUNT  NORMALIZED_CYLOMATIC_COMPLEXITY  \
0                           8          14                             0.16   
1                           6          15                             0.06   
2                           8          15                             0.06   
3                           2           6                             0.10   
4                          24          85                             0.08   
..                        ...         ...                              ...   
339                        18          34                             0.16   
340                        12          26                             0.09   
341                         8          21                             0.09   
342                         2           5                             0.18   
343                        12          24                             0.10   

     NUM_OPERANDS  NUM_OPERATORS  NUM_UNIQUE_OPERANDS  NUM_UNIQUE_OPERATORS  \
0              19             44                   15                    15   
1              51             90                   32                    27   
2              37             74                   33                    22   
3               9             14                    7                    10   
4             192            229                   71                    20   
..            ...            ...                  ...                   ...   
339            98            143                   45                    33   
340           186            263                   77                    26   
341            80            129                   41                    19   
342             4             11                    4                     7   
343            65            120                   26                    29   

     NUMBER_OF_LINES  PERCENT_COMMENTS  LOC_TOTAL Defective  
0                 32              4.00         25         N  
1                 67             39.22         32         Y  
2                 83             47.27         33         Y  
3                 20              0.00         12         N  
4                172             11.67        106         N  
..               ...               ...        ...       ...  
339               68             18.00         42         N  
340               81              7.81         59         N  
341               58              8.70         42         N  
342               11              0.00          9         N  
343               67             30.36         40         N  

[344 rows x 38 columns]
LOC_BLANK                          0
BRANCH_COUNT                       0
CALL_PAIRS                         0
LOC_CODE_AND_COMMENT               0
LOC_COMMENTS                       0
CONDITION_COUNT                    0
CYCLOMATIC_COMPLEXITY              0
CYCLOMATIC_DENSITY                 0
DECISION_COUNT                     0
DECISION_DENSITY                   0
DESIGN_COMPLEXITY                  0
DESIGN_DENSITY                     0
EDGE_COUNT                         0
ESSENTIAL_COMPLEXITY               0
ESSENTIAL_DENSITY                  0
LOC_EXECUTABLE                     0
PARAMETER_COUNT                    0
HALSTEAD_CONTENT                   0
HALSTEAD_DIFFICULTY                0
HALSTEAD_EFFORT                    0
HALSTEAD_ERROR_EST                 0
HALSTEAD_LENGTH                    0
HALSTEAD_LEVEL                     0
HALSTEAD_PROG_TIME                 0
HALSTEAD_VOLUME                    0
MAINTENANCE_SEVERITY               0
MODIFIED_CONDITION_COUNT           0
MULTIPLE_CONDITION_COUNT           0
NODE_COUNT                         0
NORMALIZED_CYLOMATIC_COMPLEXITY    0
NUM_OPERANDS                       0
NUM_OPERATORS                      0
NUM_UNIQUE_OPERANDS                0
NUM_UNIQUE_OPERATORS               0
NUMBER_OF_LINES                    0
PERCENT_COMMENTS                   0
LOC_TOTAL                          0
Defective                          0
dtype: int64
(344, 38)
样本个数：344
无缺陷个数：302
有缺陷个数：42
[0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0
 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0
 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1 0
 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0
 0 0 0 0 0 0 0 0 0 0 0]
/usr/local/lib/python3.7/site-packages/sklearn/preprocessing/data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
  return self.partial_fit(X, y)
/usr/local/lib/python3.7/site-packages/sklearn/base.py:464: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
  return self.fit(X, **fit_params).transform(X)
/usr/local/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
SVM 预测结果： [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
SVM 预测准确率： 0.9130434782608695
SVM 预测准确率： 0.9130434782608695
linear SVM 预测结果： [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
linear SVM 预测准确率： 0.8985507246376812
[17:24:55] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:516: 
Parameters: { boosting_type, subsample_freq } might not be used.

  This may not be accurate due to some parameters are only used in language bindings but
  passed down to XGBoost core.  Or some parameters are not used but slip through this
  verification. Please open an issue if you find above cases.


[0]	train-auc:0.80358	valid-auc:0.62963
Multiple eval metrics have been passed: 'valid-auc' will be used for early stopping.

Will train until valid-auc hasn't improved in 37 rounds.
[37]	train-auc:0.95874	valid-auc:0.83069
[74]	train-auc:0.96537	valid-auc:0.82011
Stopping. Best iteration:
[61]	train-auc:0.96426	valid-auc:0.83333

xgboost 预测结果: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
xgboost 预测准确率： 0.8695652173913043"""