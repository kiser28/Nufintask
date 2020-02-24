#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:37:10 2020

@author: kirillserykh
"""

import os
import operator
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats
import time

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.cluster import MiniBatchKMeans, KMeans, MeanShift
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_samples

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from copy import deepcopy
from pmdarima.arima.utils import ndiffs
from pmdarima.arima.utils import nsdiffs

def index_groupby(df, index_columns):

    ## function used for the transformation of the tuple indices created after the groupby statements:
    """ Parameters:
    df: dataframe for which groupby statement was done
    index_columns: list of columns that became indices
    """

    index_df = pd.DataFrame(df.index.tolist())
    index_df.columns = index_columns
    index_df.index = range(len(index_df))
    df.index = range(len(df))

    return pd.concat([index_df, df], axis = 1)

def diff_graphs(data, save_graph, show_graph):

    ## function that builds graphs helping to define stationarity and initial parameters of ARIMA model:
    """ Parameters:
    data: time series for which graph differencies are built
    show_graph: True if show the graph, false if not
    save_graph: True if save the graph, False if not
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.figsize':(15,12), 'figure.dpi':120, 'font.size': 8})

    fig, axes = plt.subplots(3, 3, sharex=False)
    axes[0, 0].plot(data)
    axes[0, 0].set_title('Original Series')
    axes[0, 0].format_xdata = mdates.DateFormatter('%Y-%m-%d')
    axes[0, 0].grid(True)
    plot_acf(data, ax = axes[0, 1])
    plot_pacf(data, ax=axes[0, 2])

    # 1st Differencing
    axes[1, 0].plot(data.diff())
    axes[1, 0].set_title('1st Order Differencing')
    axes[1, 0].format_xdata = mdates.DateFormatter('%Y-%m-%d')
    axes[1, 0].grid(True)
    plot_acf(data.diff().dropna(), ax=axes[1, 1])
    plot_pacf(data.diff().dropna(), ax=axes[1, 2])

    # 2nd Differencing
    axes[2, 0].plot(data.diff().diff())
    axes[2, 0].set_title('2nd Order Differencing')
    axes[2, 0].format_xdata = mdates.DateFormatter('%Y-%m-%d')
    axes[2, 0].grid(True)
    plot_acf(data.diff().diff().dropna(), ax=axes[2, 1])
    plot_pacf(data.diff().diff().dropna(), ax=axes[2, 2])
    

    if save_graph == True:
        plt.savefig(os.getcwd() + '\\' + str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")) + 'basic_diffs.png')
    if show_graph == True:
        plt.show()

def seasonal_diff_graphs(data, save_graph, show_graph, season_diff_number):

    ## function that builds graphs that help to identify seasonality:
    """ Parameters:
    data: time series for which graph differencies are built
    show_graph: True if show the graph, false if not
    save_graph: True if save the graph, False if not
    season_diff_number: number of seasonal difference to be shown
    """

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

    # Usual Differencing
    axes[0].plot(data[:], label='Original Series')
    axes[0].plot(data[:].diff(1), label='Usual Differencing')
    axes[0].set_title('Usual Differencing, period = 1')
    axes[0].legend(loc='upper right', fontsize=6)


    # Seasinal Defferencing
    axes[1].plot(data[:], label='Original Series')
    axes[1].plot(data[:].diff(season_diff_number), label='Seasonal Differencing', color='green')
    axes[1].set_title('Seasonal Differencing, period = ' + str(season_diff_number))
    plt.legend(loc='upper right', fontsize=6)

    if save_graph == True:
        # To implement: name of graph based on cluster (not really necessary right now)
        plt.savefig(os.getcwd() + '\\' + str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")) + 'season_diffs.png')
    if show_graph == True:
        plt.show()


def stationarity_tests(data):

    ## function that performs stationarity test on data:
    """ Parameters:
    data: time series for which stationarity tests are performed
    """
    return_dict = {'usual_differencing':{'ADF_test': ndiffs(data.values, test='adf'),
                                        'KPSS_test': ndiffs(data.values, test='kpss'),
                                        'PP_test': ndiffs(data.values, test='pp')},
                    'seasonal_differencing': {'Canova-Hansen': nsdiffs(data.values, m=7, max_D=31,test='ch'),
                                                'OCSB': nsdiffs(data.values, m=7, max_D=31,test='ocsb')}}
    return return_dict

def confidence_interval_values(data, alpha):

    ## supplementatry function for CI calculation:
    """ Parameters:
    data : data for which CI are calcualted
    alpha: alpha for CIs
    """
    a = np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf(1 - alpha / 2., n-1)
    return {'lower_values': a - h, 'upper_values': a + h}

def forecast_accuracy(forecast, actual):

    ## function that calcualtes the forecasting accuracy metrics:
    """ Parameters:
    forecast: forecasted values for the out-of-sample validation
    actial: actival values for the out of sample validation
    """
    mae = np.mean(np.abs(forecast - actual))
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)
    return({'mae': mae, 'rmse':rmse,'corr':corr, 'minmax':minmax})

def best_model_bic(data, test, season_number, plot_diag, save_graph):

    ## selection of the best model using the help of auto_arima function of package pdmarima:
    """ Parameters:
    data: data for which best model would be performed
    test: stationarity test selected for model Estimation
    season_number: number of seasonality to be checked
    plot_diag: True or False to plot the plot_diagnostics function
    show_graph: True if show the graph, false if not
    save_graph: True if save the graph, False if not
    """

    model = pm.auto_arima(data, start_p = 0, start_q = 0,
                      test = test, max_p = 3, max_q = 3,
                      m = season_number, d = None, seasonal = True, start_P = 0,
                      start_Q = 0, D = None, trace = True,
                      error_action = 'ignore', suppress_warnings = True,
                      stepwise = True, information_criteria = 'bic')
    if plot_diag == True:
        model.plot_diagnostics(figsize = (8,6))
        plt.savefig(os.getcwd() + '_' + str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")) + 'plot_diag.png')
    return model.to_dict()

def forecasting_metrics(data, order, seasonal_order,
                        save_graph, show_graph, testing_ratio):

    ## function for determination of forecasting metrics for the SARIMA model:
    """ Parameters:
    data: data for which model would be checked
    order: order of non-seasonal components of SARIMA model
    seasonal_order: order of seasonal components of SARIMA model
    show_graph: True if show the graph, false if not
    save_graph: True if save the graph, False if not
    testing_ratio: part of the sample to be forecasted
    
    data = df_total_forecast
    order = bic_model['order']
    seasonal_order = bic_model['seasonal_order']
    save_graph = True
    show_graph = True
    testing_ratio = testing_ratio
    
    
    """
    
    train = data[:round(len(data)*(1 - testing_ratio))]
    test = data[round(len(data)*(1 - testing_ratio)):]
    forecast_model = SARIMAX(endog = train, order = order,
                           # seasonal_order = seasonal_order,
                           initialization='approximate_diffuse',
                           enforce_stationarity = False,
                           enforce_invertability = False,
                           trend = 'c')
    fitted = forecast_model.fit(disp=0)
    fitted.summary()
        
    fc_series = fitted.forecast(len(data) - round(len(data)*(1 -testing_ratio)), alpha=0.05)
    fc_series.reset_index()
    fc_series.index = test.index
    
    lower_series = pd.Series(confidence_interval_values(fc_series, 0.05)['lower_values'], index = test.index)
    upper_series = pd.Series(confidence_interval_values(fc_series, 0.05)['upper_values'], index = test.index)

    model_name =  'SARIMA (' + ' '.join(map(str, order)) + ') x (' + ' '.join(map(str, seasonal_order)) + ')'

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals ' + model_name)
    plt.legend(loc='upper left', fontsize=8)
    if save_graph == True:
        plt.savefig(os.getcwd() + '\\' + str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")) + '_' + model_name + '_forecasts_diffs_manual.png')
    if show_graph == True:
        plt.show()
    else:
        plt.close()
    return forecast_accuracy(fc_series.values, test.as_matrix().reshape((len(fc_series.values,))))


def select_estimated_models(max_p, max_d, max_q, max_P, max_D, max_Q, season_number, season_number_ints):

    ## function for definition of all possible combinations of SARIMA parameters:
    """ Parameters:
    max_p: maximum parameter of p
    max_d: maximum parameter of d
    max_q: maximum parameter of q
    max_P: maximum parameter of P
    max_D: maximum parameter of D
    max_Q: maximum parameter of Q
    season_number: Seasonal parameter
    season_number_ints: how many times seasonal parameter multiplied with this number would be used in a model selection

    """

    p = range(1, max_p + 1)
    q = range(0, max_q + 1)
    d = range(0, max_d + 1)
    P = range(1, max_P + 1)
    D = range(0, max_D + 1)
    Q = range(0, max_Q + 1)
    S = range(season_number, season_number*(season_number_ints +1), season_number)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(P, D, Q, S))]

    params_list = list(itertools.product(pdq, seasonal_pdq))
    del_list = []
    for i in range(len(params_list)):
    ## implementing of additional restrictions to reduce the number of possible cases:
        if (params_list[i][0][0] + params_list[i][1][0] < 2) or (params_list[i][0][0] + params_list[i][1][0]) > min(max_p, max_P):
            del_list.append(params_list[i])
        if params_list[i][0][1] + params_list[i][1][1] > min(max_p, max_P):
            del_list.append(params_list[i])
        if (params_list[i][0][2] + params_list[i][1][2] < 2) or (params_list[i][0][2] + params_list[i][1][2]) > min(max_q, max_Q):
            del_list.append(params_list[i])
    params_list = list(set(params_list) - set(del_list))
    return params_list

def models_selection(data, params_list, bic_model,
                     selection_metric, testing_ratio,
                     save_graph, show_graph):

    ## function for the selection of the best model based on chosen criteria:
    """ Parameters:
    data: data for which all models would be compared
    params_list: list of parameters defined for model valuation
    bic_model: best bic model chosen for the relevant data
    testing_ratio: testing ratio for forecast validation
    selection_metric: metric based on which the selection would be done: 'rmse', 'mae', 'minmax', 'corr'
    show_graph: True if show the graph, false if not
    save_graph: True if save the graph, False if not
    
    data = df_total_forecast
    params_list = params_list
    bic_model = model_config
    testing_ratio = 0.25
    selection_metric = 'rmse'
    show_graph = False
    save_graph = False

    """
    start_time = time.time()
    models_results_brute = {}
    new_params_list = []
    for i in range(len(params_list)):
        ## try / except for invertability condition (some models don't converge)
        try:
            model_results = forecasting_metrics(data = data,
                        order = params_list[i][0],
                        seasonal_order = params_list[i][1],
                        save_graph = save_graph,
                        show_graph = show_graph,
                        testing_ratio = testing_ratio)
            new_params_list.append(params_list[i])
            models_results_brute['(' + ' '.join(map(str, params_list[i][0])) + ') x (' + ' '.join(map(str, params_list[i][1])) + ')'] = model_results
        except ValueError:
            print("The model " + str(params_list[i][0]) + "x " + str(params_list[i][1]) + "can't be estimated due to invertibility condition")
        
    print("Time for brute selection of models: --- %s seconds ---" % (time.time() - start_time))

    brute_results_df = pd.DataFrame(models_results_brute).T
    brute_results_df.index = new_params_list

    best_bic_model_results = forecasting_metrics(data = data,
            order = bic_model['order'],
            seasonal_order = bic_model['seasonal_order'],
            save_graph = save_graph,
            show_graph = show_graph,
            testing_ratio = testing_ratio)

    best_bic_model = {(bic_model['order'], bic_model['seasonal_order']):best_bic_model_results}
    best_bic_results_df = pd.DataFrame(best_bic_model).T

    total_model_results_df = pd.concat([brute_results_df,best_bic_results_df], axis = 0)
    
    ## criteria of nest model selection: 
    if selection_metric in ['rmse', 'mae', 'minmax']:
        metric_value = min(total_model_results_df[selection_metric])
        best_total_model = total_model_results_df[total_model_results_df[selection_metric] == min(total_model_results_df[selection_metric])].index[0]
    elif selection_metric in ['corr']:
        metric_value =  max(total_model_results_df[selection_metric])
        best_total_model = total_model_results_df[total_model_results_df[selection_metric] == max(total_model_results_df[selection_metric])].index[0]
    else:
        print("You've chosen the wrong model!")

    return_dict = {'best_model_params': best_total_model,
                    'metric_name': selection_metric,
                    'metric_value': metric_value}
    return return_dict

def final_forecasting(data, forecast_periods, best_model_dict,
                     save_graph, show_graph):
    
    ## function for the forecasting of the chosen model:
    """ Parameters:
    data: data for which the forecasting is done
    forecast_periods: number of periods to be forecasted
    best_model_dict: best model chosen from 2 evaluation steps
    show_graph: True if show the graph, false if not
    save_graph: True if save the graph, False if not

    """
    forecast_model = SARIMAX(endog = data, order = best_model_dict['best_model_params'][0],
                           seasonal_order = best_model_dict['best_model_params'][1],
                           enforce_stationarity = False,
                           enforce_invertability = False,
                           trend = 'c',
                           initialization='approximate_diffuse')

    model_name = 'SARIMA (' + ' '.join(map(str, best_model_dict['best_model_params'][0])) + ') x (' + ' '.join(map(str,  best_model_dict['best_model_params'][1])) + ')'
    
    ## forecasting and acquiring the results:
    results = forecast_model.fit()
    pred_uc = results.get_forecast(steps = forecast_periods)
    
    # taking CI:
    pred_ci = pred_uc.conf_int()
    
    # dates manipulation:
    dates = pd.Series(data.index)
    extra_dates = pd.Series(pd.bdate_range(dates.iloc[-1], periods = forecast_periods + 1).date).iloc[1:]

    predicted = pd.Series(pred_uc.predicted_mean)
    predicted.index = extra_dates
    pred_ci.index = extra_dates
    
    # plot:
    ax = data.plot(label='observed', figsize=(15, 8))
    predicted.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Days')
    ax.set_ylabel('Forecasted values')
    ax.set_title('Forecasted values for model ' + model_name + ':')

    plt.legend()

    if save_graph == True:
        # To implement: name of graph based on cluster (not really necessary right now)
        plt.savefig(os.getcwd() + '\\' + str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")) + '_' + model_name + 'forecast_final_plot.png')
    if show_graph == True:
        plt.show()
    
    #storing the result:
    return_dict = {'best_model_params': best_model_dict['best_model_params'],
                    'metric_name': best_model_dict['metric_name'],
                    'metric_value': best_model_dict['metric_value'],
                    'forecasted_values': predicted}
    return return_dict


