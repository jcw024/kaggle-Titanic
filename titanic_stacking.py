import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import time
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, \
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

start = time.time()

#read training data from csv file, split data for cross validation
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
PassengerId = test_df['PassengerId']
#train_df, test_df = train_test_split(train_df, test_size=0.2)

#get rid of data not helpful for predicting passenger survival
train_df = train_df.drop(['PassengerId','Name','Ticket'],axis=1)
test_df = test_df.drop(['PassengerId','Name','Ticket'],axis=1)

#visualizing relationship between Region Departed and Survival
#some NaN data in Embarked column, fill in with most common datapoint, "S"
train_df["Embarked"] = train_df["Embarked"].fillna("S")
test_df["Embarked"] = test_df["Embarked"].fillna("S")

#transform the data under 'Embarked' column into columns of their own with 1's and 0's datapoints
embark_dummies_train = pd.get_dummies(train_df['Embarked'])
embark_dummies_train.drop(['S'], axis=1, inplace=True)

embark_dummies_test = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

train_df = train_df.join(embark_dummies_train)
test_df = test_df.join(embark_dummies_test)

train_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)

#get mean and std dev of fares for survivors/non-survivors
fare_not_survived = train_df["Fare"][train_df["Survived"]==0]
fare_survived = train_df["Fare"][train_df["Survived"]==1]

average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])
train_df.fillna(train_df["Fare"].median(), inplace=True)
test_df.fillna(test_df["Fare"].median(), inplace=True)
train_df["Fare"] = train_df["Fare"].astype(int)
test_df["Fare"] = test_df["Fare"].astype(int)

#fill in NaN Age data with random values between mean +/- std dev
average_age_train = train_df["Age"].mean()
std_age_train = train_df["Age"].std()
count_nan_age_train = train_df["Age"].isnull().sum()

average_age_test = test_df["Age"].mean()
std_age_test = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

rand_train = np.random.randint(average_age_train - std_age_train, average_age_train \
                               + std_age_train, size = count_nan_age_train)
rand_test = np.random.randint(average_age_test - std_age_test, average_age_test \
                              + std_age_test, size = count_nan_age_test)
train_df.loc[np.isnan(train_df['Age']),'Age'] = rand_train
test_df.loc[np.isnan(test_df['Age']),'Age'] = rand_test

#get rid of cabin data for this analysis
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)

#simplify number of siblings and spouses into one category, "Family"
train_df['Family'] = train_df["Parch"] + train_df["SibSp"]
train_df.loc[train_df['Family'] > 0, 'Family'] = 1
train_df.loc[train_df['Family'] == 0, 'Family'] = 0

test_df['Family'] = test_df["Parch"] + test_df["SibSp"]
test_df.loc[test_df['Family'] > 0, 'Family'] = 1
test_df.loc[test_df['Family'] == 0, 'Family'] = 0

train_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)
test_df.drop(['Parch', 'SibSp'], axis=1, inplace=True)

#visualize relationship between sex/child and survival
#reformat the data from 'sex' column into their own columns with 1's and 0's
def get_person(passenger):
    age, sex = passenger
    return 'Child' if age < 16 else sex

train_df['Person'] = train_df[["Age", "Sex"]].apply(get_person, axis=1)
test_df['Person'] = test_df[["Age", "Sex"]].apply(get_person, axis=1)
train_df.drop(["Sex"], axis=1, inplace=True)
test_df.drop(["Sex"], axis=1, inplace=True)

person_dummies_train = pd.get_dummies(train_df['Person'])
person_dummies_train.columns = ['Child', 'Female', 'Male']
person_dummies_train.drop(['Male'], axis=1, inplace=True)

person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train_df = train_df.join(person_dummies_train)
test_df = test_df.join(person_dummies_test)
train_df.drop(["Person"], axis=1, inplace=True)
test_df.drop(["Person"], axis=1, inplace=True)

#reformat data under Pclass into their own columns for economic class
pclass_dummies_train = pd.get_dummies(train_df['Pclass'])
pclass_dummies_train.columns = ['Upper Class', 'Middle Class', 'Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Upper Class', 'Middle Class', 'Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)

train_df = train_df.join(pclass_dummies_train)
test_df = test_df.join(pclass_dummies_test)

#useful parameters for later
ntrain = train_df.shape[0]
ntest = test_df.shape[0]
NFOLDS = 5
SEED = 8
kf = KFold(n_splits=NFOLDS, random_state=SEED)

#class to extend Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train,y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances_(self,x,y):
        return self.clf.fit(x,y).feature_importances_

#out of fold predictions
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test.reshape(-1,1)

#setup classifier parameters
#random forest
rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': True,
        #'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
        }

#extra trees
et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        #'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
        }

#adaboost
ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
        }

#gradient boosting
gb_params = {
        'n_estimators': 500,
        #'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
        }

#support vector classifier
svc_params = {
        'kernel': 'linear',
        'C': 0.025
        }


#create model objects
now1 = time.time()
print('at runtime {}: '+'creating model objects').format(now1 - start)
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

y_train = train_df['Survived'].ravel()
train_df = train_df.drop(['Survived'], axis=1)
x_train = train_df.values
#test_df = test_df.drop(['Survived'],axis=1)
x_test = test_df.values
now2 = time.time()
print('created model objects in {}s').format(now2 - now1)

print('getting out of fold predictions')
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)
now3 = time.time()
print('got out of fold predictions in {}s').format(now3 - now2)

print('getting feature importances')
rf_features = rf.feature_importances_(x_train,y_train)
et_features = et.feature_importances_(x_train,y_train)
ada_features = ada.feature_importances_(x_train,y_train)
gb_features = gb.feature_importances_(x_train,y_train)
now4 = time.time()
print('got feature importances in {}s').format(now4 - now3)

#plotting/visualizing feature importances
cols = train_df.columns.values
feature_dataframe = pd.DataFrame({'features':cols,
    'Random Forest feature importances': rf_features,
    'Extra Trees feature importances': et_features,
    'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })

print "generating feature importance plots"
fig1 = sns.factorplot('features','Random Forest feature importances', data=feature_dataframe, kind='bar',size=4,aspect=2.2)
fig1.fig.suptitle('Random Forest feature importances')
fig1.savefig("fig1.png")
fig2 = sns.factorplot('features','Extra Trees feature importances', data=feature_dataframe, kind='bar',size=4,aspect=2.2)
fig2.fig.suptitle('Extra Trees feature importances')
fig2.savefig("fig2.png")
fig3 = sns.factorplot('features','AdaBoost feature importances', data=feature_dataframe, kind='bar',size=4,aspect=2.2)
fig3.fig.suptitle('AdaBoost feature importances')
fig3.savefig("fig3.png")
fig4 = sns.factorplot('features','Gradient Boost feature importances', data=feature_dataframe, kind='bar',size=4,aspect=2.2)
fig4.fig.suptitle('Gradient Boost feature importances')
fig4.savefig("fig4.png")

#create new column with average of values
feature_dataframe['mean'] = feature_dataframe.mean(axis=1)
fig5 = sns.factorplot('features','mean', data=feature_dataframe, kind='bar',size=4,aspect=2.2)
fig5.savefig("fig5.png")
now5 = time.time()
print("feature importances plotted in {}s").format(now5 - now4)
print('feature importances saved to fig1,2,3,4,5.png')

base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
    'ExtraTrees': et_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientBoost': gb_oof_train.ravel()
    })

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train),axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test),axis=1)

#fitting and training second level learning model
print "fitting and training second level learning model"
gbm = xgb.XGBClassifier(
        #learning_rate = 0.02
        n_estimators = 2000,
        max_depth = 4,
        min_child_weight = 2,
        gamma = 0.9,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary:logistic',
        nthread = -1,
        scale_pos_weight = 1).fit(x_train, y_train)
predictions = gbm.predict(x_test)
trainpred = gbm.predict(x_train)
now6 = time.time()
print("fit and trained second level learning model in {}s").format(now6-now5)

submission = pd.DataFrame({'PassengerId': PassengerId,
    'Survived': predictions})
submission.to_csv("submission.csv", index=False)
print('predictions on test file saved to submission.csv')
print "training data prediction accuracy: ", accuracy_score(trainpred, y_train)
now7 = time.time()
print("Total elapsed time: {}s").format(now7 - start)
