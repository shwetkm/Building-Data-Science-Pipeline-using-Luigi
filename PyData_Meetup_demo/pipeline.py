from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import luigi
import json



""" 
	Luigi Task 1  : Load Data
	Inp : Path to the data to be loaded
	Out : Raw data stored in local target
	Logic : This task will bring data from source(it maybe some DB as well) and store at desired target location
"""
class LoadData(luigi.Task):
	filepath = luigi.Parameter()
	def requires(self):
		return []

	def output(self):
		return luigi.LocalTarget('Raw/raw_{}'.format(self.filepath[6:]))

	def run(self):
		df = pd.read_csv(self.filepath)
		df.to_csv(self.output().path, index=False)



""" 
	Luigi Task 2  : Aggregate Data
	Inp : Raw Data from different sources
	Out : Aggregated data stored in local target
	Logic : This task aggregates data from different sources
"""
class AggregateData(luigi.Task):
	source1 = luigi.Parameter()
 	source2 = luigi.Parameter()

 	def requires(self):
		return [LoadData(filepath =self.source1),LoadData(filepath =self.source2)]

	def output(self):
		return luigi.LocalTarget('Aggregated/agg_data.csv')

	def run(self):
		df1 = pd.read_csv(self.input()[0].path)
		df2 = pd.read_csv(self.input()[1].path)
		df = df1.merge(df2,on = 'character_id',how = 'outer')
		df.to_csv(self.output().path, index=False)



""" 
	Luigi Task 3  : Pre-process Data
	Inp : Aggregated Data
	Out : Pre-processed Data
	Logic : Remove given columns, Drop missing value rows
"""
class PreprocessData(luigi.Task):
	remove_f = luigi.ListParameter()
 	source1 = luigi.Parameter()
 	source2 = luigi.Parameter()

 	def requires(self):
		return [AggregateData(source1 =self.source1,source2 = self.source2)]

	def output(self):
		return [luigi.LocalTarget('Train/train.csv'),luigi.LocalTarget('Test/test.csv')]

	def run(self):
		df = pd.read_csv(self.input()[0].path)
		df = df.drop([item for item in self.remove_f],axis=1)
		df = df.dropna(axis = 0, how = 'any')
		df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)
		df_train.to_csv(self.output()[0].path, index=False)
		df_test.to_csv(self.output()[1].path, index=False)



""" 
	Luigi Task 4  : Linear Regression
	Inp : Pre-processed Data
	Out : Model Score and Model pickle
	Logic : Fits Linear Regression to the the train data and evaluates model on test data
"""
class Linear(luigi.Task):
	source1 = luigi.Parameter()
	source2 = luigi.Parameter()
	remove_f = luigi.Parameter()
	target = luigi.Parameter()

	def requires(self):
		return [PreprocessData(source1 =self.source1,source2 = self.source2,remove_f = self.remove_f)]

	def output(self):
		return [luigi.LocalTarget('Score/linearscore.json'),luigi.LocalTarget('Pickl/linearmodel.pkl')]

	def run(self):
		train,test = pd.read_csv(self.input()[0][0].path), pd.read_csv(self.input()[0][1].path)
		clf = LinearRegression()
		clf.fit(train.drop(self.target,axis=1),train[self.target])
		acc = metrics.mean_squared_error(test[self.target], clf.predict(test.drop(self.target,axis=1)))
		result = {'mse':acc}
		with self.output()[0].open('w') as f:
			json.dump(result, f)
		
		joblib.dump(clf, self.output()[1].path)



"""
	Luigi Task 5  : Random Forest Regression
	Inp : Pre-processed Data
	Out : Model Score and Model pickle
	Logic : Fits Random Forest Regression to the the train data and evaluates model on test data 
"""
class RandomForest(luigi.Task):
	source1 = luigi.Parameter()
	source2 = luigi.Parameter()
	remove_f = luigi.Parameter()
	target = luigi.Parameter()

	def requires(self):
		return [PreprocessData(source1 =self.source1,source2 = self.source2,remove_f = self.remove_f)]

	def output(self):
		return [luigi.LocalTarget('Score/rfscore.json'),luigi.LocalTarget('Pickl/rfmodel.pkl')]

	def run(self):
		train, test = pd.read_csv(self.input()[0][0].path), pd.read_csv(self.input()[0][1].path)
		clf = RandomForestRegressor()
		clf.fit(train.drop(self.target,axis=1),train[self.target])
		acc = metrics.mean_squared_error(test[self.target], clf.predict(test.drop(self.target,axis=1)))
		result = {'mse':acc}
		with self.output()[0].open('w') as f:
			json.dump(result, f)
		joblib.dump(clf, self.output()[1].path)



""" 
	Luigi Task 6  : Modelling
	Inp : Model Scores, Model Pickles
	Out : Prediction on given data
	Logic : Selects the best model and predicts scores for given user data
"""
class Modelling(luigi.Task):
	source1 = luigi.Parameter()
	source2 = luigi.Parameter()
	remove_f = luigi.ListParameter()
	target = luigi.Parameter()
	pred_path = luigi.Parameter()

	def requires(self):
		return [PreprocessData(source1 =self.source1,source2 = self.source2,remove_f = self.remove_f),\
				Linear(source1 =self.source1,source2 = self.source2,remove_f = self.remove_f,target=self.target),\
				RandomForest(source1 =self.source1,source2 = self.source2,remove_f = self.remove_f,target=self.target)]

	def output(self):
		return luigi.LocalTarget('Prediction/predicted.csv')

	def run(self):
		score_lr,score_rf = json.load(self.input()[1][0].open('r')), json.load(self.input()[2][0].open('r'))

		# Model Selection
		if score_lr['mse']<score_rf['mse']:
			pkl = joblib.load(self.input()[1][1].open('r'))
		else:
			pkl = joblib.load(self.input()[2][1].open('r'))
		
		pred = pd.read_csv(self.pred_path)
		pred = pred.drop([item for item in self.remove_f],axis=1)
		prediction = pd.DataFrame(pkl.predict(pred))
		prediction.to_csv(self.output().path, index=False)

if __name__ == '__main__':
	luigi.run()

# python pipeline.py Modelling --source1 Train/training.csv --source2 Train/character.csv --remove-f '["user_id","character_id","user_character_id"]' --target score --pred-path Test/final_test.csv