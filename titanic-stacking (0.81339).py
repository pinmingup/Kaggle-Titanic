# # # 导入包
import numpy as np
import pandas as pd
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import csv

from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.neural_network import MLPClassifier  # NN
from sklearn.tree import DecisionTreeClassifier  # 决策树
from xgboost import XGBClassifier  # xgboost
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # 二次判别分析（QDA）
from sklearn import svm  # 支持向量机SVM
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯

# # 集成算法
from sklearn.ensemble import BaggingClassifier  # bagging
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import GradientBoostingClassifier  # GBDT
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost

# # 回归算法
from sklearn.linear_model import LogisticRegression  # 最小二乘回归OLS
from sklearn.linear_model import Ridge  # 岭回归
from sklearn.kernel_ridge import KernelRidge  # 核岭回归
from sklearn.svm import SVR  # 支持向量机回归
from sklearn.linear_model import Lasso  # 套索回归
from sklearn.linear_model import ElasticNet  # 弹性网络回归
from sklearn.linear_model import BayesianRidge  # 贝叶斯回归

# # 数据读入
train_df = pd.read_csv('C:/Users/dell/Desktop/python-tc/titanic_data/titanic_train.csv')
test_df = pd.read_csv('C:/Users/dell/Desktop/python-tc/titanic_data/titanic_test.csv')
# # 数据处理
combine_df = pd.concat([train_df, test_df])
# Title
combine_df['Title'] = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
combine_df['Title'] = combine_df['Title'].replace(['Don', 'Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Dr'], 'Mr')
combine_df['Title'] = combine_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
combine_df['Title'] = combine_df['Title'].replace(['the Countess', 'Mme', 'Lady', 'Dr'], 'Mrs')
df = pd.get_dummies(combine_df['Title'], prefix='Title')
combine_df = pd.concat([combine_df, df], axis=1)
# Name_length
combine_df['Name_Len'] = combine_df['Name'].apply(lambda x: len(x))
combine_df['Name_Len'] = pd.qcut(combine_df['Name_Len'], 5)
# Dead_female_family & Survive_male_family
combine_df['Surname'] = combine_df['Name'].apply(lambda x: x.split(',')[0])
dead_female_surname = list(set(combine_df[(combine_df.Sex == 'female') & (combine_df.Age >= 12) &
                                          (combine_df.Survived == 0) & ((combine_df.Parch > 0) |
                                                                        (combine_df.SibSp > 0))]['Surname'].values))
survive_male_surname = list(set(combine_df[(combine_df.Sex == 'male') & (combine_df.Age >= 12) &
                              (combine_df.Survived == 1) & ((combine_df.Parch > 0) |
                                                            (combine_df.SibSp > 0))]['Surname'].values))
combine_df['Dead_female_family'] = np.where(combine_df['Surname'].isin(dead_female_surname), 0, 1)
combine_df['Survive_male_family'] = np.where(combine_df['Surname'].isin(survive_male_surname), 0, 1)
combine_df = combine_df.drop(['Name', 'Surname'], axis=1)
# Age & isChild
group = combine_df.groupby(['Title', 'Pclass'])['Age']
combine_df['Age'] = group.transform(lambda x: x.fillna(x.mean()))
combine_df = combine_df.drop('Title', axis=1)
combine_df['IsChild'] = np.where(combine_df['Age'] <= 12, 1, 0)
combine_df['Age'] = pd.cut(combine_df['Age'], 5)
combine_df = combine_df.drop('Age', axis=1)
# ticket
combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))

combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1', '2', 'P']), 1, 0)
combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['A', 'W', '3', '7']), 1, 0)
combine_df = combine_df.drop(['Ticket', 'Ticket_Lett'], axis=1)
# Embarked
# combine_df = combine_df.drop('Embarked',axis=1)
combine_df.Embarked = combine_df.Embarked.fillna('S')
df = pd.get_dummies(combine_df['Embarked'], prefix='Embarked')
combine_df = pd.concat([combine_df, df], axis=1).drop('Embarked', axis=1)
# FamilySize
combine_df['FamilySize'] = np.where(combine_df['SibSp']+combine_df['Parch'] == 0, 'Alone',
                                    np.where(combine_df['SibSp']+combine_df['Parch'] <= 3, 'Small', 'Big'))
df = pd.get_dummies(combine_df['FamilySize'], prefix='FamilySize')
combine_df = pd.concat([combine_df, df], axis=1).drop(['SibSp', 'Parch', 'FamilySize'], axis=1)

# Cabin
combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(), 0, 1)
combine_df = combine_df.drop('Cabin', axis=1)

# PClass
df = pd.get_dummies(combine_df['Pclass'], prefix='Pclass')
combine_df = pd.concat([combine_df, df], axis=1).drop('Pclass', axis=1)

# Sex
df = pd.get_dummies(combine_df['Sex'], prefix='Sex')
combine_df = pd.concat([combine_df, df], axis=1).drop('Sex', axis=1)

# Fare
combine_df['Fare'].fillna(combine_df['Fare'].dropna().mean(), inplace=True)
combine_df['Low_Fare'] = np.where(combine_df['Fare'] <= 8.662, 1, 0)
combine_df['High_Fare'] = np.where(combine_df['Fare'] >= 26, 1, 0)
combine_df = combine_df.drop('Fare', axis=1)

features = combine_df.drop(["PassengerId", "Survived"], axis=1).columns
le = LabelEncoder()
for feature in features:
    le = le.fit(combine_df[feature])
    combine_df[feature] = le.transform(combine_df[feature])

X_all = combine_df.iloc[:891, :].drop(["PassengerId", "Survived"], axis=1)
Y_all = combine_df.iloc[:891, :]["Survived"]
# print(Y_all.head())
X_test = combine_df.iloc[891:, :].drop(["PassengerId", "Survived"], axis=1)
# print(X_test.shape)
# print(X_all.info())


# # 第一层模型
bag = BaggingClassifier(random_state=2018)
mlp = MLPClassifier(random_state=2018)
lr = LogisticRegression(random_state=2018)
svc = SVC(random_state=2018)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=2018)
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255}, random_state=2018)
gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=3, random_state=2018)
xgb = XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.01, random_state=2018)
clfs = [bag, mlp, lr, svc, knn, dt, rf, gbdt, xgb]

ntrain = train_df.shape[0]  # 891
ntest = test_df.shape[0]  # 418
kf = KFold(n_splits=5, random_state=2018)


# stacking model func
def get_oof(clf, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))
    for i, (train_index, test_index)in enumerate(kf.split(X_train)):
        kf_X_train = X_train.iloc[train_index]
        kf_y_train = y_train.iloc[train_index]
        kf_X_test = X_train.iloc[test_index]
        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index] = clf.predict(kf_X_test)
        oof_test_skf[i, :] = clf.predict(X_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


train_final = pd.DataFrame()
test_final = pd.DataFrame()
i = 0
for clf in clfs:
    i += 1
    oof_train, oof_test = get_oof(clf, X_all, Y_all, X_test)
    oof_train = pd.DataFrame(oof_train, columns=['x'+str(i)])
    oof_test = pd.DataFrame(oof_test, columns=['x'+str(i)])
    train_final = pd.concat([train_final, oof_train], axis=1)
    test_final = pd.concat([test_final, oof_test], axis=1)
new_train = pd.concat([train_final, train_df['Survived']], axis=1)
# print(new_train)
new_test = test_final

# # 第二层模型
lr = LogisticRegression()
new_clf = xgb
# data
fea = []
for i in range(1, len(clfs)+1):
    fea.append('x'+str(i))
# print(fea)
train_x = new_train[fea]
train_y = new_train['Survived']
x1, x2, y1, y2 = train_test_split(train_x, train_y, test_size=0.5, random_state=2018)
new_clf.fit(x1, y1)
plot_importance(new_clf)
# plt.show()
resx2 = new_clf.predict(x2)
print(mean_squared_error(resx2, y2))
# new_clf.fit(train_x, train_y)
res = new_clf.predict(new_test[fea])
# # 写入文件
# submission = pd.DataFrame({
#         "PassengerId": test_df["PassengerId"],
#         "Survived": res
#     })
# submission.to_csv(r'titanic1203.csv', index=False)