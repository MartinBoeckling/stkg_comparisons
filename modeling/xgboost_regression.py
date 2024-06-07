import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
import json
import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from multiprocessing import cpu_count
import wandb

class regressionModel:
    """_summary_
    """
    def __init__(self, validation: str, dataPath: str, testDate: str, resume: bool) -> None:
        """_summary_

        Args:
            validation (str): _description_
            dataPath (str): _description_
            testDate (str): _description_
            resume (bool): _description_
        """
        self.dataPath = Path(dataPath)
        self.validationDate = validation
        # create directory for use case
        self.loggingPath = Path('modeling').joinpath(self.dataPath.stem)
        self.loggingPath.mkdir(exist_ok=True, parents=True)
        self.resume = resume
        self.validation = validation
        self.cpuCore = int(cpu_count()/2)
        # check if input is eligeble for data processing
        # check if dataPath input is a file
        assert self.dataPath.is_file(), f'{self.dataPath} does not link to a file'
        # check if testDate input can be interpreted as date
        assert not pd.isna(pd.to_datetime(testDate, errors='coerce')), f'{testDate} is not an eligible date'
        # assign test date as class variable
        self.testDate = testDate
        trainData, testData = self.dataPreparation()
        if validation:
            parameterSettings = self.parameterTuning(trainData)
            # oversample complete training set for training
        else:
            parameterSettings = {}
        self.modelTraining(trainData, testData, parameterSettings)

    def dataPreparation(self) -> tuple:
        """_summary_

        Returns:
            tuple: _description_
        """
        print('Data Preprocessing')
        # read file into dataframe
        data = pd.read_parquet(self.dataPath)
        data = data.rename({'date': 'DATE'}, axis=1)
        # split data into train and testset based on specified date
        # create train dataframe which is under specified date
        trainData = data[data['DATE'] < self.testDate]
        # create test dataframe which is over specified date
        testData = data[data['DATE'] >= self.testDate]
        # extract target column
        trainDataY = trainData.pop('price')
        # Drop unnecessary columns
        trainDataX = trainData.drop(columns=['listing_id', 'id', 'DATE'], axis=1, errors='ignore')
        # extract price column as testdata target
        testDataY = testData.pop('price')
        # Drop Date and ID column
        testDataX = testData.drop(columns=['listing_id', 'id', 'date', 'DATE'], axis=1, errors='ignore')
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
        numericFeatures = testDataX.select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']).columns
        # select columns with categorical dtype
        categoricFeatures = testDataX.select_dtypes(
            include=['object']).columns
        # construct column transformer object to apply pipelines to each column
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numericTransformer, numericFeatures),
                ('cat', categoricTransformer, categoricFeatures)],
            n_jobs=self.cpuCore, verbose=True, remainder='passthrough')
        # apply column transformer to train and test data
        consideredColumns = testDataX.columns
        consideredData = data[consideredColumns]
        preprocessor.fit(consideredData)
        trainDataX = preprocessor.transform(trainDataX)
        testDataX = preprocessor.transform(testDataX)
        if self.validation:
            trainDataDate = pd.DataFrame(
                trainDataDate, columns=['date'])
            trainDataDate = trainDataDate['date'].astype(object)
            trainDataX = pd.DataFrame.sparse.from_spmatrix(trainDataX)
            trainDataDate = trainDataDate.reset_index(drop=True)
            trainDataX = pd.concat([trainDataX, trainDataDate], axis=1)
        # return trainData and testdata X and Y dataframe in tuple format
        return (trainDataX, trainDataY), (testDataX, testDataY)
    
    def parameterRoutineCV(self, minChildWeight: float, maxDepth: float, subsample: float,
                          colsampleBytree: float, colsampleBylevel: float, gamma: float, numberEstimators: float) -> float:
        """_summary_

        Args:
            minChildWeight (float): _description_
            maxDepth (float): _description_
            subsample (float): _description_
            colsampleBytree (float): _description_
            colsampleBylevel (float): _description_
            gamma (float): _description_
            numberEstimators (float): _description_

        Returns:
            float: _description_
        """
        xgbCl = xgb.XGBRegressor(seed=15, n_jobs=self.cpuCore, tree_method='hist',
                                 min_child_weight=minChildWeight,
                                 max_depth = int(maxDepth),
                                 subsample=subsample,
                                 colsample_bytree = colsampleBytree,
                                 colsample_bylevel= colsampleBylevel,
                                 gamma = gamma,
                                 n_estimators = int(numberEstimators))
        # create time series cross validation object
        timeSeriesCV = TimeSeriesSplit(n_splits=4)
        # calculate cross validation score
        # iterate over times series cross validation
        trainDataX = self.dataTrainX
        trainDataX['date'] = pd.to_datetime(trainDataX['date'])
        # sort DATE column to enable time series split feature
        trainDataX = trainDataX.sort_values(by='date')
        cvScores = []
        for trainIndex, testIndex in timeSeriesCV.split(trainDataX):
            # select dataframe rows based on extracted index
            trainX, trainY = self.dataTrainX.iloc[trainIndex], self.dataTrainY.iloc[trainIndex]
            testX, testY = self.dataTrainX.iloc[testIndex], self.dataTrainY.iloc[testIndex]
            trainX = trainX.drop(['date', 'index'], axis=1, errors='ignore')
            testX = testX.drop(['date', 'index'], axis=1, errors='ignore')
            xgbCl.fit(trainX, trainY)
            predClass = xgbCl.predict(testX)
            MAPE = mean_absolute_percentage_error(testY, predClass)
            cvScores.append(-MAPE)
        cvScoresNumpy = np.array(cvScores)
        return cvScoresNumpy.mean()
    

    def parameterTuning(self, dataTrain: pd.DataFrame) -> dict:
        """specify bayesian search cross validation with the following specifications
            - estimator: specified extra gradient boosting classifier
            - search_spaces: defined optimization area for hyperparameter tuning
            - cv: Specifying split for cross validation with 5 splits
            - scoring: optimization function based on f1_marco-score optimization
            - verbose: Output while optimizing
            - n_jobs: Parallel jobs to be used for optimization using 2 jobs
            - n_iter: Iteration for optimization
            - refit: Set to false as only parameter settings need to be extracted

        Args:
            dataTrain (pd.DataFrame): _description_

        Returns:
            dict: _description_
        """
        print('Parameter tuning')
        # extract dataframes from train and test data tuples
        self.dataTrainX = dataTrain[0]
        self.dataTrainY = dataTrain[1]
        optimizer = BayesianOptimization(f=self.parameterRoutineCV,
                                        pbounds={
                                            'minChildWeight': (0, 100),
                                            'maxDepth': (0, 50),
                                            'subsample': (0.01, 1.0),
                                            'colsampleBytree': (0.01, 1.0),
                                            'colsampleBylevel': (0.01, 1.0),
                                            'gamma': (0, 50),
                                            'numberEstimators': (50, 1000)
                                        },
                                        verbose=2,
                                        random_state=14)
        if self.resume:
            load_logs(optimizer, logs=[f'{self.loggingPath}/logs.json'])
            with open(f'{self.loggingPath}/logs.json') as loggingFile:
                loggingFiles = [json.loads(jsonObj) for jsonObj in loggingFile]
            iterationSteps = 50 - len(loggingFiles) + 10
            initPoints = 0
        else:
            iterationSteps = 10
            initPoints = 50
        logger = JSONLogger(path=f"{self.loggingPath}/logs.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(n_iter=iterationSteps, init_points=initPoints)
        print(f'Best parameter & score: {optimizer.max}')
        dataIterationPerformance = pd.json_normalize(optimizer.res)
        dataIterationPerformance.to_csv(f'{self.loggingPath}/runPerformance.csv', index=False)
        parameterNames = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'max_depth', 'min_child_weight', 'n_estimators', 'scale_pos_weight', 'subsample']
        parameterCombination = dict(zip(parameterNames, optimizer.max['params'].values()))
        parameterCombination['max_depth'] = int(parameterCombination['max_depth'])
        parameterCombination['n_estimators'] = int(parameterCombination['n_estimators'])
        return parameterCombination
    

    def modelTraining(self, trainData: tuple, testData: tuple, parametersettings: dict) -> None:
        """_summary_

        Args:
            trainData (tuple): _description_
            testData (tuple): _description_
            parametersettings (dict): _description_
        """
        print('Model training')
        # extract dataframes from train and test data tuples
        dataTrainX = trainData[0]
        dataTrainY = trainData[1]
        dataTestX = testData[0]
        dataTestY = testData[1]
        file_name = self.dataPath.stem
        tags = file_name.split("_")
        if self.validation:
            dataTrainX = dataTrainX.drop(labels='date', axis=1, errors='ignore')
        # specify extra gradient boosting classifier
        # estimate scale_pos_weight value
        self.run = wandb.init(
            project='stkg_comparison_runs',
            tags=tags,
            reinit=True
        )
        xgbRegressor = xgb.XGBRegressor(**parametersettings, seed=15, n_jobs=self.cpuCore, tree_method='exact')
        # fit specified model to training data
        xgbRegressor.fit(dataTrainX, dataTrainY)
        # perform prediction on test dataset with trained model
        predValues = xgbRegressor.predict(dataTestX)
        self.run.summary["MAE"] = mean_absolute_error(dataTestY, predValues)
        self.run.summary["MDAE"] = median_absolute_error(dataTestY, predValues)
        self.run.summary["MAPE"] = mean_absolute_percentage_error(dataTestY, predValues)
        self.run.summary["R2"] = r2_score(dataTestY, predValues)
        self.run.finish()


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'
    # initialize the command line argparser
    parser = argparse.ArgumentParser(description='XGBoost argument parameters')
    # add validation argument parser
    parser.add_argument('-v', '--validation', default=False, action='store_true',
    help="use parameter if grid parameter search should be performed")
    parser.add_argument('-r', '--resume', default=False, action='store_true',
    help="use parameter if grid parameter search should be resumed")
    # add path argument parser
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='string value to data path')
    # add date argument parser
    parser.add_argument('-d', '--date', type=str, required=True,
    help='date value for train test split', default='2017-04-01')
    # store parser arguments in args variable
    args = parser.parse_args()
    # Pass arguments to class function to perform xgboosting
    model = regressionModel(validation=args.validation, dataPath=args.path, testDate=args.date, resume=args.resume)