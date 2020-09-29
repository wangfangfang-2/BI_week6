import pandas as pd
from lightgbm import LGBMClassifier
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
from sklearn.metrics import accuracy_score
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
from sklearn.pipeline import make_pipeline
import smote_variants as sv



input_data_path = r'/Users/zhangyuxi/Documents/B_I_/kaggle'
output_data_path = r'/Users/zhangyuxi/Documents/B_I_/kaggle'
encoder = 'LabelEncoder'  # 'OneHotEncoder' 'LabelEncoder'
is_PolynomialFeatures = True
is_predict = True
ia_GridSearchCV = False
algorithm = 'lightgbm'  # 'lightgbm', 'RF', 'LR', 'SVC'

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")


category_list = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
    'MaritalStatus', 'Over18', 'OverTime'
]
Scaler_list = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]


def main():
    # 数据读取处理
    X, y, df_test, category_list = data_processing()
    # 特征处理
    X, category_list, feature_engineering_dict = feature_engineering(
        X, category_list)
    # 模型训练
    output_model = model_processing(X, y, category_list)
    # 预测测试集
    if is_predict:
        predict_df = predict_processing(feature_engineering_dict, df_test,
                                        category_list, output_model)
        predict_df.to_excel('{}/predict_dfV13.xlsx'.format(output_data_path),
                            index=False)


def data_processing():
    # 训练数据读取处理
    df_train = pd.read_csv(input_data_path + '/train.csv',
                           dtype={'user_id': str})
    y = df_train[['Attrition']]
    y_le = LabelEncoder().fit(['Yes', 'No'])
    y['Attrition'] = y_le.transform(y['Attrition'])
    X = df_train.drop('Attrition', axis=1)
    X = X.drop('user_id', axis=1)
    # 测试数据读取处理
    df_test = pd.read_csv(input_data_path + '/test.csv')
    return X, y, df_test, category_list


def feature_engineering(df_train, category_list):
    feature_engineering_dict = {}
    encoder_dict = {}

    # 标准化处理
    Scaler_dict = {}
    Scaler_dict['max'] = {}
    Scaler_dict['min'] = {}
    for col_name in Scaler_list:
        data_max = df_train[col_name].max()
        data_min = df_train[col_name].min()
        df_train[col_name] = (df_train[col_name] - data_min) / (data_max -
                                                                data_min)
        Scaler_dict['max'][col_name] = data_max
        Scaler_dict['min'][col_name] = data_min
    feature_engineering_dict['Scaler_dict'] = Scaler_dict
    # OneHotEncoder还是LabelEncoder编码
    if encoder == 'OneHotEncoder':
        encoder_dict['le_dict'] = {}
        encoder_dict['oe_dict'] = {}
        data_OneHotEncoder = []
        for col_name in category_list:
            le = LabelEncoder().fit(df_train[col_name])
            encoder_dict['le_dict'][col_name] = le
            df_train[col_name] = le.transform(df_train[col_name])
            oe = OneHotEncoder().fit(df_train[[col_name]])
            encoder_dict['oe_dict'][col_name] = oe
            df = oe.transform(df_train[[col_name]]).toarray().tolist()
            column_df = pd.DataFrame(df,
                                     columns=oe.get_feature_names([col_name]))
            data_OneHotEncoder.append(column_df)
        df_train_sub = pd.concat(data_OneHotEncoder, axis=1)
        df_train = pd.concat([df_train, df_train_sub], axis=1)
    elif encoder == 'LabelEncoder':
        encoder_dict = {}
        for col_name in category_list:
            le = LabelEncoder().fit(df_train[col_name])
            encoder_dict[col_name] = le
            df_train[col_name] = le.transform(df_train[col_name])
    # df_train[category_list] = df_train[category_list].astype('category')
    feature_engineering_dict['encoder_dict'] = encoder_dict
    # 是否需要交叉特征
    if is_PolynomialFeatures:
        polynomialFeatures_list = df_train.columns.tolist()
        poly = PolynomialFeatures(3, interaction_only=True, include_bias=False)
        df_train = poly.fit_transform(df_train)
        df_train = pd.DataFrame(df_train,
                                columns=poly.get_feature_names(
                                    input_features=polynomialFeatures_list))
        feature_engineering_dict['PolynomialFeatures'] = poly
    for col_name in Scaler_list:
        df_train['{}_sqrt'.format(col_name)] = np.sqrt(df_train[col_name])
    return df_train, category_list, feature_engineering_dict


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
            plt.savefig('{output_data_path}/plot_importance_Lgb.jpg'.format(
                output_data_path=output_data_path))
            plt.close()
            # 输出最好的分数
            print('best_score_ is {}'.format(output_model.best_score_))
            # 输出最佳迭代次数
            best_iteration = output_model.best_iteration_
            print('best_iteration_ is {}'.format(best_iteration))
            # 输出metric图
            _, ax = plt.subplots(figsize=(20, 10))
            plot_metric(output_model, ax=ax, metric='auc')
            plt.savefig('{output_data_path}/plot_metric_Lgb.jpg'.format(
                output_data_path=output_data_path))
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
            plt.savefig(R"{}/train_confusion_LGB.jpg".format(output_data_path))
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
            plt.savefig(R"{}/test_confusion_LGB.jpg".format(output_data_path))
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
            plt.savefig('{output_data_path}/auc_LGB.jpg'.format(
                output_data_path=output_data_path))
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
            plt.savefig('{output_data_path}/auc_RF.jpg'.format(
                output_data_path=output_data_path))
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
            plt.savefig('{output_data_path}/auc_LR.jpg'.format(
                output_data_path=output_data_path))
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
            plt.savefig('{output_data_path}/auc_SVC.jpg'.format(
                output_data_path=output_data_path))
            plt.close()
    return output_model


def predict_processing(feature_engineering_dict, df_test, category_list,
                       output_model):
    encoder_dict = feature_engineering_dict['encoder_dict']
    predict_data = df_test.copy()
    product_code_df = predict_data[['user_id']]
    predict_data = predict_data.drop('user_id', axis=1)
    Scaler_dict = feature_engineering_dict['Scaler_dict']
    for col_name in Scaler_list:
        data_max = Scaler_dict['max'][col_name]
        data_min = Scaler_dict['min'][col_name]
        predict_data[col_name] = (predict_data[col_name] -
                                  data_min) / (data_max - data_min)
    if encoder == 'OneHotEncoder':
        data_OneHotEncoder = pd.DataFrame()
        for col_name in category_list:
            le = encoder_dict['le_dict'][col_name]
            predict_data[col_name] = le.transform(predict_data[col_name])
            oe = encoder_dict['oe_dict'][col_name]
            # predict_data[col_name] = oe.transform(predict_data[[col_name]])
            df = oe.transform(predict_data[[col_name]]).toarray().tolist()
            column_df = pd.DataFrame(df,
                                     columns=oe.get_feature_names([col_name]))
            data_OneHotEncoder = pd.concat([data_OneHotEncoder, column_df],
                                           axis=1)
        predict_data = data_OneHotEncoder
    elif encoder == 'LabelEncoder':
        for col_name in category_list:
            le = encoder_dict[col_name]
            predict_data[col_name] = predict_data[col_name].map(
                lambda s: '<unknown>' if s not in le.classes_ else s)
            le_classes = le.classes_.tolist()
            bisect.insort_left(le_classes, '<unknown>')
            le.classes_ = le_classes
            predict_data[col_name] = le.transform(predict_data[col_name])

    if is_PolynomialFeatures:
        polynomialFeatures_list = predict_data.columns.tolist()
        poly = feature_engineering_dict['PolynomialFeatures']
        predict_data = poly.transform(predict_data)
        predict_data = pd.DataFrame(
            predict_data,
            columns=poly.get_feature_names(
                input_features=polynomialFeatures_list))
    for col_name in Scaler_list:
        predict_data['{}_sqrt'.format(col_name)] = np.sqrt(predict_data[col_name])

    # predict_data[category_list] = predict_data[category_list].astype(
    #     'category')
    predict_data.to_excel('{}/predict_data_X.xlsx'.format(output_data_path))
    predict_y = output_model.predict(predict_data)
    predict_y_df = pd.DataFrame(predict_y, columns=['predict_label'])
    predict_df = pd.concat([product_code_df, predict_y_df], axis=1)
    y_pred_pro = output_model.predict_proba(predict_data)
    y_pred_pro_df = pd.DataFrame(y_pred_pro, columns=['N_pro', 'Y_pro'])
    predict_df = pd.concat([predict_df, y_pred_pro_df], axis=1)
    print(predict_df)
    return predict_df


if __name__ == "__main__":
    main()
