"""
Description: The present module contains all the processes required to produce the forecast. It uses functions
from the HDC module to generate the full historical data necessary to make a forecast and then it makes and
individual forecast per each combination of Channel and Language. Finally, it saves the updated historical
data and forecasts into the database

Author: Leonardo Jul Camargo

Department: Basic-Fit Customer Service Workforce Management

Date: 2023-03-01

Version: 1.1
"""
#Import custom libraries
from CodeLibrary import helper_data_postprocessing, helper_data_preparation, helper_sarimax, helper_query_data_dwh
import hdc as h
import seamly

#Import time and date libraries
import datetime
from datetime import  timedelta, date

#Import basic libraries
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys
import yaml

pd.options.mode.chained_assignment = None

# Determine environment
with open('CodeLibrary/environment.yml') as file:
    settings_yaml = yaml.load(file, Loader=yaml.FullLoader)
environment = settings_yaml.get('environment_CC')
environment_push_data = settings_yaml.get('environment_push_data_CC')

# Determine environment (if passed through commandline)
"""
if len(sys.argv) > 1:
    environment = str(sys.argv[1])
    environment_push_data = str(sys.argv[2])
"""

def all_cols_x_plus_x():
    """
    Generates a HDC and then uses it to produce the X+4 forecast for all its columns 
    and saves the HDC and the predictions to two different tables in the database.

    Example:

        .. code-block:: python

            import pandas as pd
            test_data = {'Daily_volumes':[10,15,84,32,56,78,94,321,45,97,64,13,64,97,163,6,716,5,6,71,6,4,6],
                        'Daily_volumes_2':[15,84,32,56,78,94,31,45,97,64,13,64,97,13,6,76,5,6,71,6,4,6,10]}
            date_generated = pd.date_range(start='1/1/2022',end='1/23/2022')
            test_dataframe = pd.DataFrame(test_data, index=date_generated)
            all_cols_x_plus_x(test_dataframe)
    """
    #Generate HDC
    current_time = str(datetime.datetime.now())

    missed_chats = seamly.get_all_missed_chats()

    df = h.make_hybrid_hdc(missed_chats)

    hdc = df.copy()

    hdc_save = hdc.copy()

    hdc_save.drop(hdc_save.index[:-90], axis = 0, inplace = True)

    hdc_recoded = h.recode(hdc_save,'Actuals')

    hdc_recoded.reset_index(inplace=True)

    #Saving the historical data
    helper_query_data_dwh.post_data_table(environment_push_data,
                                        'dbo.TEST_HDC',
                                        hdc_recoded.columns,
                                        hdc_recoded,
                                        current_time)

    #Get the config from the yaml file
    languages, channels = get_config(type = 'forecast')
    
    with open('daily_forecast.yml') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)

    #Make a prediction for each Channel + Language combination
    preds_list = pd.DataFrame()

    for l in languages:

        for c in channels:

            if l not in config.get('forecast').get('data_sources').get(c).get('languages'): continue

            oc_full_data = hdc[[c + " " + l]]

            result = predict_x_plus_x(oc_full_data)

            preds_list[c + " " + l] = result

    preds_list.index.name = 'date'

    #Temporary file for investigation
    #preds_list.to_csv("predictions_daily.csv")

    preds_list_recoded = h.recode(preds_list, 'Prediction')

    preds_list_recoded.reset_index(inplace=True)

    #Save predictions to the DB
    post_data = helper_query_data_dwh.post_data_table(environment_push_data,
                                                      'dbo.TEST_NewCustomeCareForecast2',
                                                       preds_list_recoded.columns,
                                                       preds_list_recoded,
                                                       current_time)
    
    print(f"""\033[92m\033[1mSuccess!! X + 5 Forecast complete!!
        Forecasted Channels\033[0m:

        {preds_list.columns}
        """)

    return

def predict_x_plus_x(oc_full_data: pd.DataFrame):
    """
    Predicts the next five weeks from the last date of a given one-column dataframe.

    Args:
        oc_full_data (pd.DataFrame): A one-column dataframe with time series data to be prediceted

    Returns:
        prediction_sx (pd.DataFrame): A single-column, one-row dataframe with the daily prediction for the next 5 weeks.

    Example:

        .. code-block:: python

            import pandas as pd
            test_data = {'Daily_volumes':[1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3]}
            date_generated = pd.date_range(start='4/8/2022',end='4/30/2022')
            test_dataframe = pd.DataFrame(test_data, index=date_generated)
            predict_x_plus_x(test_dataframe)
    """
    oc_full_data_copy= oc_full_data.copy()

    #Pre-process training data
    country = oc_full_data_copy.columns[0]

    country = country[-2:].upper()

    #Excluding debtors from outlier detection
    if "debtors" not in oc_full_data_copy.columns[0].lower():
        oc_full_data_copy = helper_data_preparation.remove_outliers(oc_full_data_copy,28)

    if "phone" in oc_full_data_copy.columns[0].lower() or "chat" in oc_full_data_copy.columns[0].lower():
        no_holidays_oc_full_data = helper_data_preparation.rem_holidays(oc_full_data_copy, country)
    else:
        no_holidays_oc_full_data = helper_data_preparation.adj_holidays(oc_full_data_copy, country)

    #Fit training data to model
    sx = helper_sarimax.sarimax_fit(no_holidays_oc_full_data)
    
    #Make prediction for X+4
    prediction_sx = helper_sarimax.sarimax_predictv2(sarimax_model = sx, 
                                                      start = no_holidays_oc_full_data.index[len(no_holidays_oc_full_data)-1] + timedelta(days = 1),   #4 weeks after the last day in the dataframe
                                                      end = no_holidays_oc_full_data.index[len(no_holidays_oc_full_data)-1]+timedelta(weeks = 5),       #5 weeks after the last day in the dataframe
                                                      col_name = no_holidays_oc_full_data.columns[0])
    
    #Post process it
    if "phone" in no_holidays_oc_full_data.columns[0].lower() or "chat" in no_holidays_oc_full_data.columns[0].lower(): 

        prediction_sx = helper_data_postprocessing.adjust_mean(prediction_sx, no_holidays_oc_full_data)
        prediction_sx = helper_data_postprocessing.add_holidays(prediction_sx, country)

    else:
        helper_data_postprocessing.adj_holidays(oc_full_data_copy,prediction_sx, country)
        prediction_sx = helper_data_postprocessing.adjust_mean(prediction_sx, no_holidays_oc_full_data)

    helper_data_postprocessing.trim_negatives(prediction_sx)

    prediction_sx.sort_index(ascending=True, inplace=True)

    prediction_sx.fillna(0,inplace=True)
 
    return prediction_sx.astype(dtype = 'int')

def get_config(type: str, file: str = 'daily_forecast.yml'):
    """
    Gets the necessary configuration to run the script from a YAML file.

    Args:
        type (str): Type of configuration to be retrieved,
        can be 'hdc' or 'forecast'
        file(str): file path and name for the yaml file

    Returns:
        languages (tuple): Languages to be used
        channels (tuple): Channels to be used

    Example:

        .. code-block:: python

            get_config('config_file.yml')
    """
    with open(file) as f:

        config = yaml.load(f, Loader=yaml.FullLoader)

    languages = config.get(type).get('languages')

    channels = config.get(type).get('channels')

    return languages, channels