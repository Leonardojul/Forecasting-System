"""
Program: Customer Service Forecasting System Insights Collector

Description: This script collects the business insights from the forecast_insights.xlxs
file and transforms the data inside it to then load it to the database so that it can
be available to be presented in the Forecasting Dashboard.

Author: Leonardo Jul Camargo

Department: Basic-Fit Customer Service Workforce Management

Date: 2023-03-01

Version: 1.0
"""
#Import custom libraries
from CodeLibrary import helper_query_data_dwh as hqd

#Import basic libraries
import pandas as pd
import yaml
import datetime
import pyodbc

#Label columns from excel file
fi_dict = {0:"Language",
           1:"Channel",
           2:"Description",
           3:"Amount",
           4:"Affected_date"}

#Read yaml file to get all available channels and languages
with open('daily_forecast.yml') as f:

    config = yaml.load(f, Loader=yaml.FullLoader)

channels = config.get('hdc').get('channels')

languages = config.get('hdc').get('languages')

replace_data = {'Channel':channels, 'Language':languages}

indexes = list(fi_dict.keys())

names = list(fi_dict.values())

#Load the data
fi = pd.read_excel("forecast_insights.xlsx", sheet_name= "Forecast_insights" ,engine = 'openpyxl')

fi = fi.iloc[:,indexes]

fi.columns=(names)

explode_list = ['Language','Channel']

#Explode all packed languages and channels from single lines to multiple lines
for i in explode_list:

    replace_string = ','.join(str(s) for s in replace_data[i])

    fi[i] = fi[i].fillna(replace_string)

    fi[i] = fi[i].str.upper().str.split(',')

    fi = fi.explode(i)

    fi[i] = fi[i].str.replace(" ", "")

fi['Channel'] = fi['Channel'].str.capitalize()

fi['Amount'] = fi['Amount'].astype(dtype= 'object')

fi.reset_index(drop= True, inplace= True)

fi.fillna(0,inplace=True)

environment = 'DEV'

table_name = 'dbo.Forecast_insights'

user, pwd, server, db, driver = hqd.get_connection_properties(environment)

db_conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={db};UID={user};PWD={pwd}')

cursor = db_conn.cursor()

cursor.execute(f"TRUNCATE TABLE {table_name}")

db_conn.commit()
cursor.close()

post_data = hqd.post_data_table(environment,
                                table_name,
                                fi.columns,
                                fi,
                                datetime.date.today())

print("Success!")