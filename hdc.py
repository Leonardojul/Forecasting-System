"""
Description: The present module gathers historical data from every relevant channel and subchannel, and then saves this aggregated
data to a table in the database so that it can be later on used for reporting or forecasting.

Author: Leonardo Jul Camargo

Department: Basic-Fit Customer Service Workforce Management

Date: 2023-03-01

Version: 1.1
"""
#Import custom libraries
from CodeLibrary import helper_query_data_dwh as hqd
import jobs as job

#Import time and date libraries
from datetime import date, timedelta, datetime
import datetime
import pytz

#Import basic libraries
import sys
import yaml
import pandas as pd
import textwrap

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

def make_hybrid_hdc(missed_chat_hdc: pd.DataFrame):
    """
    Builds full hdc by combining the data from SalesForce extracted from the SQL server
    and includes missed chats extracted from the Seamly API

    Args:
        missed_chat_hdc (pd.DataFrame): DataFrame constructed with the get_all_missed_chats
        from the seamly module

    Returns:
        hdc (pd.DataFrame): The combination of the two hdcs in a single dataframe.    
    """

    #Get the most up to date data from the SQL server
    hdc, languages, channels = construct_hdc()
 
    for l in languages:

        hdc[f'Chat {l}'] += missed_chat_hdc[f'missed_chat_{str(l).lower()}']

    #Create totals column
    hdc['Totals'] = hdc.sum(axis=1, numeric_only=True)

    hdc = hdc.asfreq(freq = 'D')

    hdc.fillna(0,inplace=True)

    hdc.astype(int)

    return hdc

def construct_hdc():
    """
    Queries the SQL DB to build a dataframe with daily volumes for all selected channels 
    in the configuration YML file. Run time is approximately 10 minutes.

    Returns:
        hdc (pd.DataFrame): Dataframe with a compilation of all data queried

    """
    #Check the configuration to select the right queries
    languages, channels = job.get_config(type='hdc')

    channels_list = []
    with open('daily_forecast.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
    #Query the DB for each channel and save results in dataframe
    for l in languages:

        for c in channels:

            if 'DB' not in config.get('hdc').get('data_sources').get(c).get('sources'): continue

            if l not in config.get('hdc').get('data_sources').get(c).get('languages'): continue

            channel = c + " " + l

            sql_file = (f"SQL queries/{channel}.sql")

            query = create_query_string(sql_file)

            channels_list.append(get_channel(query, channel))

    #Consolidate dataframe with all the queries
    hdc = pd.concat(channels_list,axis=1)

    hdc.fillna(0, inplace=True)

    return hdc, languages, channels

def create_query_string(sql_file: str, start_date: str = str(date.today() -timedelta(days = 180)), end_date:str = str(date.today() -timedelta(days = 1))):
    """
    Modifies an existing SQL query to select the results within a time range.
    Note: this function relies on a SQL query having an already existing WHERE clause,
    which filters on a time range, for example:

    .. code-block:: sql

        SELECT *
        FROM table_t
        WHERE [Date] BETWEEN '2022-1-1' AND '2022-1-31'

    Args:
        sql_file (str): File name and path of the base SQL file
        start_date (str): Desired date to use for the filtering (oldest date)
        end_date (str): Most recent date of the time range to be used
        Note: if no dates are given as arguments, the start date will be 2022-1-2 and the end
        date will be always yesterday

    Returns:
        temp_query (str): Formatted SQL query with the filtering dates modified by the user

    Example:

        .. code-block:: python

            create_query_string('some_sql_file.sql', start_date = '2018-1-1', end_date = '2021-12-31')
    """
    with open(sql_file) as f_in:
        #Determine time difference between pipeline and server time
        dt = datetime.datetime.today()-timedelta(days=1)

        if is_dst(dt):
            adjusted_time = " 22:00:00"
            time_offset = 2
        
        else:
            adjusted_time = " 23:00:00"
            time_offset = 1

        lines = f_in.read()

        #Remove common leading whitespace from all lines
        query = textwrap.dedent("""{}""".format(lines))

        #Find the CreatedDate clause filter
        ds_pos = query.find('SET @date_start')

        #Replace start date
        start_date_pos_a = query.find('\'',ds_pos) + 1

        start_date_pos_b = query.find('\'',start_date_pos_a)

        temp_query = query[:start_date_pos_a] + start_date + adjusted_time + query[start_date_pos_b:]

        #Find the EndDate clause filter   
        end_date_pos_a = temp_query.find('\'',start_date_pos_b + 1) + 1

        end_date_pos_b = temp_query.find('\'',end_date_pos_a)

        #Replace end date
        temp_query = temp_query[:end_date_pos_a] + end_date + adjusted_time + query[end_date_pos_b:]

        #Find hours_to_add clause and set it accordingly
        if temp_query.find("@hours_to_add") != -1:

            time_offset_pos_a = temp_query.find('= ',end_date_pos_b + 1) +1

            time_offset_pos_b = temp_query.find(';',time_offset_pos_a)

            temp_query = temp_query[:time_offset_pos_a] + str(time_offset) + temp_query[time_offset_pos_b:]

        return temp_query
  
def get_channel(query: str, channel: str):
    """
    Sends the entered query to the chosen database and retrieves the query output,
    returning it as a pandas DataFrame.
    Note: this function only works with single column returning queries.

    Args:
        query (str): SQL query to be sent to server
        channel (str): Channel name to be retrieved, this will only be used to name the
        column of the generated DataFrame

    Returns:
        df (pd.DataFrame): Dataframe with the results of the query

    Example:

        .. code-block:: python

            df = get_channel('SELECT * FROM table', 'Chat_NL')
    """

    df = hqd.get_data(environment, query, "")

    df.columns = (['date', channel])

    df.set_index('date', inplace = True)

    df.index = pd.to_datetime(df.index)

    df.fillna(0,inplace=True)

    df = df.astype('int')

    return df

def recode(df: pd.DataFrame, column_name: str):
    """
    Takes a dataframe containing a time index and multiple columns and stacks the
    columns together categorically.

    Args:

        df: pd.DataFrame
        column_name: Name to be given to the new column

    Returns:
        df_recoded (pd.DataFrame): A new dataframe with the columns stacked into ['Channel']
        and ['Language']

    Example:

    .. code-block:: python

        >>>df

        date       Email_NL Email_FR Chat_NL Chat_FR
        2022-01-01    100     120      110     105
        2022-01-02     90     110      105     125

        >>>df_recoded = recode(df, 'Actuals')
        >>>df_recoded

        date        Channel  Language Actuals
        2022-01-01    Email     NL      100
        2022-01-01    Email     FR      120
        2022-01-01    Chat      NL      110
        2022-01-01    Chat      FR      105
        2022-01-02    Email     NL       90
        2022-01-02    Email     FR      110
        2022-01-02    Chat      NL      105
        2022-01-02    Chat      FR      125

    """
    df_recoded = pd.DataFrame(df.stack())

    df_recoded.reset_index(inplace=True)

    df_recoded.set_index('date', inplace= True)

    df_recoded.columns = (['channel',column_name])

    df_recoded[['Channel','language']] = df_recoded['channel'].str.split(' ', expand = True)

    df_recoded['Language'] = df_recoded['language'].str.upper()

    df_recoded.drop(['channel','language'], axis=1,inplace=True)

    return df_recoded

def is_dst(dt: datetime.datetime):
    """
    Checks whether a date falls in daylight saving time.

    Args:
        dt(datetime.datetime): The date to be checked

    Returns:
        (bool): If true, it means the entered date is in summer time

    Example:

        .. code-block:: python

            import datetime
            dt = datetime.datetime(2022,9,1)
            print(is_dst(dt))
            Result: "True"
    """
    
    time_zone = pytz.timezone("Europe/Amsterdam")

    aware_dt = time_zone.localize(dt)

    return aware_dt.dst() != datetime.timedelta(0,0)