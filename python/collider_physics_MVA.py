#To  run  this macro you need python3, xgboost, matplotlib, scikit-learn, pandas, numpy etc
#that are provided in the environment that we setup during the class  with
#conda activate myenv.

import time

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sb
import xgboost as xgb


import ROOT

ROOT.gROOT.SetBatch(False)

#Before starting, let's define some pplotting functions
def makeplot(df, ylabel, var_name):
  ax = df[df[ylabel]==0].plot.hist(column=[var_name], figsize=(10, 8), bins = 50, density = True, alpha=0.7)
  ax.set_yscale('log')
  df[df[ylabel]==1].plot.hist(column=[var_name], figsize=(10, 8), bins = 50, density = True, alpha=0.7, ax=ax)
  ax.legend(labels=[var_name+' bkg', var_name+' sig'])
  plt.savefig(var_name+".png") 

def plotcorr(df, ylabel):
  fig, axs = plt.subplots(ncols=2, figsize=(18, 6), dpi=300)
  # calculate the correlation matrix on backgorund examples
  corr_bkg = df[df[ylabel]==0].corr()
  # calculate the correlation matrix on backgorund examples
  corr_sig = df[df[ylabel]==1].corr()
  # Colors
  cmap = sb.diverging_palette(500, 10, as_cmap=True)
  # plot the heatmap
  axs[0].set_title('Background')
  sb.heatmap(corr_bkg, annot=True, cmap=cmap, ax=axs[0])
  # plot the heatmap
  axs[1].set_title('Signal')
  sb.heatmap(corr_sig, annot=True, cmap=cmap, ax=axs[1])
  plt.savefig("corr_matrix.png")

def plot_overtrainingchecks(model, X_test, y_test, X_train, y_train, model_name):
  y_pred_test = model.predict(X_test)
  y_pred_train = model.predict(X_train)

  y_pred_test_sig = y_pred_test[ y_test[:]==1] 
  y_pred_test_bkg = y_pred_test[ y_test[:]==0] 

  y_pred_train_sig = y_pred_train[ y_train[:]==1] 
  y_pred_train_bkg = y_pred_train[ y_train[:]==0] 

  #Kolmogorov-Smirnov test of distributions:
  print("KS - signal: ", ks_2samp(y_pred_test_sig, y_pred_train_sig))
  print("KS - background: ", ks_2samp(y_pred_test_bkg, y_pred_train_bkg))
  
  plt.clf()
  plt.figure(figsize=(6.6,6.6))
  plt.hist(y_pred_test_sig, label="sig_test", bins=50, range=(0.0, 1.0), histtype='step', density=True)
  plt.hist(y_pred_train_sig, label="sig_train", bins=50, range=(0.0, 1.0), histtype='step', density=True)

  plt.hist(y_pred_test_bkg, label="bkg_test", bins=50, range=(0.0, 1.0), histtype='step', density=True)
  plt.hist(y_pred_train_bkg, label="bkg_train", bins=50, range=(0.0, 1.0), histtype='step', density=True)
  
  #plt.yscale('log')
  plt.xlabel('bdt score', fontsize=14)
  plt.ylabel('Events', fontsize=14)
  plt.legend(loc=2)
  plt.savefig(model_name+"_bdt.png")

  fpr_test, tpr_test, _ = roc_curve(y_test.ravel(), y_pred_test.ravel())
  fpr_train, tpr_train, _ = roc_curve(y_train.ravel(), y_pred_train.ravel())

  plt.clf()
  plt.plot(tpr_test, 1-fpr_test, label="test AUC {:.2f}".format(roc_auc_score(y_test,y_pred_test)))
  plt.plot(tpr_train, 1-fpr_train, label="train AUC {:.2f}".format(roc_auc_score(y_train,y_pred_train)))
  
  plt.legend(loc=3)
  plt.xlabel('Signal efficiency')
  plt.ylabel('Background Rejection')
  plt.savefig(model_name+"_roc.png")

def main():
  #open the input ROOT files
  rdf_bkg = ROOT.RDataFrame("T", "../notebooks/data/MELA_variables_bkg_200k_1_5Tev_DLF.root")
  rdf_sig = ROOT.RDataFrame("T", "../notebooks/data/MELA_variables_sig_50k_1_5Tev_DLF.root")
  
  #apply some basic preselections using RDataFrame
  rdf_bkg_presel = rdf_bkg.Filter('mH<150')
  rdf_sig_presel = rdf_sig.Filter('mH<150')
  
  print("background: passed ", str(rdf_bkg_presel.Count().GetValue()), " events over total ", str(rdf_bkg.Count().GetValue()))
  print("signal: passed ", str(rdf_sig_presel.Count().GetValue()), " events over total ", str(rdf_sig.Count().GetValue()))
  
  #Data preparation for machine learning methods
  features = ['mZ1', 'mZ2', 'cosTheta_star', 'cosTheta_1', 'cosTheta_2', 'phi', 'phi_1'] #to be used in the training
  
  #labels
  rdf_bkg_presel = rdf_bkg_presel.Define('Y', str(0))
  rdf_sig_presel = rdf_sig_presel.Define('Y', str(1))
  
  #Convert from ROOT to numpy data format
  np_bkg = rdf_bkg_presel.AsNumpy()
  np_sig = rdf_sig_presel.AsNumpy()
  
  df = pd.DataFrame(np_bkg)
  df = pd.concat([df, pd.DataFrame(np_sig)], ignore_index=True)
  
  print('Printing dataframe:')
  print(df)
  
  #plotting in pandas dataframe
  for var_name in features:
    makeplot(df, 'Y', var_name)
  
  #plotting correlation matrix
  plotcorr(df, 'Y')
  
  #separate features from labels
  y = df['Y'].values
  X = df[[col for col in features]]
  
  #Split into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)
  
  #Training our first BDT model with no tuning of the parameters
  print("Training on %i examples with %i features"%X_train.shape)
  
  #Use default parameters and train on full dataset
  mymodel = xgb.sklearn.XGBRegressor(objective='binary:logistic',
                                   max_depth        = 7,
                                   learning_rate    = 0.07,
                                   n_estimators     = 1000,
                                   eval_metric= "auc" #'error',
                                   )
  #Train and time classifier
  start_time = time.time()
  mymodel.fit(X_train, y_train)
  run_time = time.time() - start_time
  
  #Make Predictions
  print("Predicting on %i examples with %i features\n"%X_test.shape)
  y_pred= mymodel.predict(X_test)
  
  #Print Results
  print("Model Accuracy: {:.2f}%".format(100*mymodel.score(X_test, y_test)))
  print("The AUC score is {:.2f}".format(roc_auc_score(y_test,y_pred)))
  print("Run time: {:.2f} sec\n\n".format(run_time))
  
  #Plot feature importance 
  fig=plt.figure(figsize=(12,9))
  xgb.plot_importance(mymodel, ax=plt.gca())
  fig.subplots_adjust(left=0.4)  #
  fig.savefig('feature_importance.png')
  
  #Perform overtraining checks
  plot_overtrainingchecks(mymodel, X_test, y_test, X_train, y_train, "my_first_model")
  
  #Optimizing XGBoost
  cv_params = {'max_depth': [3,4,6,8], 'min_child_weight': [1,3,5], 'learning_rate':[0.01,0.8]}
  model = xgb.sklearn.XGBRegressor(objective='binary:logistic',
                                   n_estimators = 300,
                                   eval_metric= "auc" #'error',
                                   )
  opt_XGBRegressor = GridSearchCV(model,
                                  cv_params,
                                  cv = 5, 
                                  n_jobs = -1, verbose=True, error_score='raise')
  
  opt_XGBRegressor.fit(X_train, y_train)
  
  print('################### Optimisation Results ###################')
  print(opt_XGBRegressor.best_score_)
  print(opt_XGBRegressor.best_params_)
  
  #Print scores
  print('The optimal score on training set is {:0.3f}'.format(opt_XGBRegressor.best_score_))
  
  #Find optimal parameters
  print('The optimal parameters for the classifier are:')
  print(opt_XGBRegressor.best_params_)
  
  #Fit performance on the test set
  XGBRegressor_final=opt_XGBRegressor.best_estimator_
  y_pred_final=XGBRegressor_final.predict(X_test)
  print("Model Accuray with optimal parameters: {:.2f}%".format(100*XGBRegressor_final.score(X_test, y_test)))
  print("The  AUC score is {:.2f}".format(roc_auc_score(y_test,y_pred_final)))
  
  #Let's now add Regularisation parameters
  mymodel2 = xgb.sklearn.XGBRegressor(objective='binary:logistic',
                                   max_depth        = 8,
                                   learning_rate    = 0.04,
                                   n_estimators     = 1000,
                                   subsample        = 0.8,
                                   colsample_bytree = 0.9,
                                   min_child_weight = 1,
                                   gamma            = 10,
                                   reg_alpha        = 10,
                                   reg_lambda       = 5,
                                   early_stopping_rounds = 10,
                                   eval_metric= 'auc',
                                   )
  
  #Train and time classifier
  start_time = time.time()
  #Note: with earli stopping you now need the validation set!
  mymodel2.fit(X_train, y_train,
              verbose = True,
              eval_set=[(X_train, y_train), (X_test, y_test)]
  )
  
  run_time = time.time() - start_time
  
  print('We now introduce regularisation parameters!')
  #Make Predictions
  print("Predicting on %i examples with %i features\n"%X_test.shape)
  y_pred= mymodel2.predict(X_test)
  
  #Print Results
  print("Model Accuracy: {:.2f}%".format(100*mymodel2.score(X_test, y_test)))
  print("The AUC score is {:.2f}".format(roc_auc_score(y_test,y_pred)))
  print("Run time: {:.2f} sec\n\n".format(run_time))
  
  #Re-run overtraining checks
  plot_overtrainingchecks(mymodel2, X_test, y_test, X_train, y_train, "model_with_regularisation")

  #Then: exercise! Look at the notebook for hints

if __name__ == "__main__":
    main()
