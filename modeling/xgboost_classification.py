''''
Title:
Extra Gradient Boosting Script

Description:
This script uses the extra gradient boosting algorithm to predict wildfires. The
hyperparameters are optimized using bayes search cross validation implemented by 
the scikit-learn optimization package. The optimization is performed on a f1-score.
The datasets are highly imbalanced, therefore a random oversampling for the minority
class is performed. The random over sampling is implemented by the imblearn package.

Input:
    - dataPath: Path of dataset for use case in format dir/.../file
    - testDate: Date where split is performed

Output:
    - Optimal parameter combination
    - Score development over time
    - Classification report implemented by scikit-learn

'''
# import packages
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import wandb


class modelPrediction:
    def __init__(self, data_path, testDate):
        # transform datafile path into pathlib object
        self.data_path = Path(data_path)
        # create directory for use case

        # check if input is eligeble for data processing
        # check if dataPath input is a file
        assert self.data_path.is_file(), f'{self.data_path} does not link to a file'
        # check if testDate input can be interpreted as date
        assert not pd.isna(pd.to_datetime(testDate, errors='coerce')), f'{testDate} is not an eligible date'
        # assign test date as class variable
        self.testDate = testDate
        # perform data preprocessing for training and test data
        trainData, testData = self.dataPreprocess()
        # perform training based on train and test dataset and parametersettings
        self.modelTraining(trainData, testData)
        #self.modelExplanation()

    def oversampleRows(self, trainDataX, trainDataY):
        # Drop geometry and ID column
        trainDataX['DATE'] = trainDataX['DATE'].astype(object)
        # 
        trainDataX.columns = trainDataX.columns.astype(str)
        roseSampling = RandomOverSampler(random_state=15)
        # resample data with specified strategy
        trainDataX, trainDataY = roseSampling.fit_resample(trainDataX, trainDataY)
        # reorder dataframes to support timeseriessplit
        trainDataY = pd.DataFrame(trainDataY, columns=['WILDFIRE'])
        # unify datasets to one dataframe
        trainData = pd.concat([trainDataX, trainDataY], axis=1)
        # change datatype to datetime to support sorting of column
        trainData['DATE'] = pd.to_datetime(trainData['DATE'])
        # sort DATE column to enable time series split feature
        trainData = trainData.sort_values(by='DATE')
        # split class from unified dataframe
        trainDataY = trainData.pop('WILDFIRE')
        # Drop Date column
        trainDataX = trainData.drop(columns=['DATE'], axis=1, errors='ignore')
        return trainDataX, trainDataY
    
    def dataPreprocess(self):
        print('Data Preprocessing')
        # read file into dataframe
        data = pd.read_parquet(self.data_path)
        # transform DATE column into datetime
        data['DATE'] = pd.to_datetime(data['DATE'])
        # check if column osmCluster is present in dataframe
        if 'osmCluster' in data.columns:
            # change datatype of column osmCluster to categorical data type
            data = data.astype({'osmCluster':'object'})
        # split data into train and testset based on specified date
        # create train dataframe which is under specified date
        print(self.testDate)
        print(data['DATE'])
        trainData = data[data['DATE'] < self.testDate]
        # create test dataframe which is over specified date
        testData = data[data['DATE'] >= self.testDate]
        # extract wildfire column as target
        trainDataY = trainData.pop('WILDFIRE')
        # Drop geometry and ID column
        trainDataX = trainData.drop(columns=['ID', 'geometry', 'YEAR'], axis=1, errors='ignore')
        trainDataX, trainDataY = self.oversampleRows(trainDataX, trainDataY)
        # extract Wildfire column as testdata target
        testDataY = testData.pop('WILDFIRE')
        # Drop Date and ID column
        testDataX = testData.drop(columns=['DATE', 'ID', 'geometry', 'YEAR'], axis=1, errors='ignore')
        self.testData = testDataX
        # create preprocessing pipeline for numerical and categorical data
        # create numerical transformer pipeline
        numericTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        # create categorical transformer pipeline
        categoricTransformer = Pipeline(steps=[
            ('encoder', OneHotEncoder())
        ])
        # select columns with numerical dtypes
        numericFeatures = trainDataX.select_dtypes(include=[int, float]).columns
        # select columns with categorical dtype
        categoricFeatures = trainDataX.select_dtypes(include=['object']).columns
        # construct column transformer object to apply pipelines to each column
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numericTransformer, numericFeatures),
                ('cat', categoricTransformer, categoricFeatures)],
            n_jobs=-5, verbose=True, remainder='passthrough')
        # apply column transformer to train and test data
        preprocessor.fit(trainDataX)
        trainDataX = preprocessor.transform(trainDataX)
        testDataX = preprocessor.transform(testDataX)
        # return trainData and testdata X and Y dataframe in tuple format
        return (trainDataX, trainDataY), (testDataX, testDataY)

    def modelTraining(self, trainData, testData):
        file_name = self.data_path.stem
        tags = file_name.split("_")
        self.run = wandb.init(
            project='stkg_comparison_run',
            tags=tags,
            reinit=True
        )
        print('Model training')
        # extract dataframes from train and test data tuples
        dataTrainX = trainData[0]
        dataTrainY = trainData[1]
        dataTestX = testData[0]
        dataTestY = testData[1]
        timeSeriesCV = TimeSeriesSplit(n_splits=5)
        # specify extra gradient boosting classifier
        xgbCl = xgb.XGBClassifier(objective="binary:logistic", seed=15, n_jobs=-5, tree_method='exact')
        predClass =  xgbCl.fit(X= dataTrainX, y=dataTrainY)
        # fit specified model to training data
        xgbCl.fit(dataTrainX, dataTrainY)
        self.model = xgbCl
        # perform prediction on test dataset with trained model
        predClass = xgbCl.predict(dataTestX)
        # calculate probability for AUC calculation
        predProb = xgbCl.predict_proba(dataTestX)[:,1]
        print(f'Feature imporance:{xgbCl.feature_importances_}')
        # print confusion matrix of classification
        print(f'Confusion matrix:\n{confusion_matrix(dataTestY, predClass)}')
        # print AUC metric
        aucScore = roc_auc_score(dataTestY, predProb, multi_class='ovo')
        print(f'AUC Score:\n{aucScore}')
        # print classification report
        print(f'Model score:\n{classification_report(dataTestY, predClass)}')
        # save roc_curve
        # extract false positive and true positive value series
        fp, tp, _ = roc_curve(dataTestY,  predProb)
        # construct roc curve plot
        # # initialize matplotlib figure
        # plt.figure()
        # # add roc curve to matplotlib figure
        # plt.plot(fp, tp, label=f"ROC Curve (AUC={aucScore})", color='dimgray', lw=2)
        # # add line regarding random performance
        # plt.plot([0, 1], [0, 1], color="darkgrey", lw=2, label=f"Random guess", linestyle="--")
        # # add limitations to x- and y-axis
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # # add labels to x-axis and y-axis
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.legend(loc="lower right")
        dataTestY = dataTestY.to_numpy()
        # Log the ROC curve to W&B
        dataTestYROC = dataTestY
        predProb2D = np.array([1-predProb, predProb])
        d = 0
        while len(predProb2D.T) > 10000:
            d +=1
            predProb2D = predProb2D[::1, ::d]
            dataTestYROC = dataTestYROC[::d]

        self.run.summary["auc"] = aucScore
        self.run.summary["F1-Score"] = f1_score(dataTestY, predClass)
        self.run.summary["Precision"] = precision_score(dataTestY, predClass)
        self.run.summary["Recall"] = recall_score(dataTestY, predClass)
        self.run.summary["conf_mat"] = wandb.sklearn.plot_confusion_matrix(dataTestY, predClass, ['No wildfire', 'Wildfire'])
        self.run.log({"ROC_Curve" : wandb.plot.roc_curve(dataTestYROC, predProb2D.T, labels=['No wildfire', 'wildfire'], classes_to_plot=[1])})
        self.run.finish()


if __name__ == '__main__':
    os.environ['WANDB_START_METHOD'] = 'thread'
    pd.options.mode.chained_assignment = None  # default='warn'
    # initialize the command line argparser
    parser = argparse.ArgumentParser(description='XGBoost argument parameters')
    # add validation argument parser
    # add path argument parser
    parser.add_argument('-p', '--path', type=str, required=True,
    help='string value to data path')
    # add date argument parser
    parser.add_argument('-d', '--date', type=str, required=True,
    help='date value for train test split', default='2020-01-01')
    # store parser arguments in args variable
    args = parser.parse_args()
    # Pass arguments to class function to perform xgboosting
    model = modelPrediction(data_path=args.path, testDate=args.date)
