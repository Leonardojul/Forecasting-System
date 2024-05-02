import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import holidays
import matplotlib.pyplot as plt

from .helper_sarimax import sarimax_predictv2, sarimax_fit

def last_day_of_month(any_day: int):
    """
    Helper function for the createPeriods. Defines the last day of the month

    Args:
        any_day: a date (mostly the start date) in the month that we would like to find the last day of the month for.
    Returns:
        datetime.date: the date of the last day of the month
    Exaple:

        .. code-block:: python

            tr_start, tr_end, pred_start, pred_end = createPeriods('2018-01-01', '2022-06-01')
            today = datetime.date.today()
            last_day_current = last_day_of_month(today).day #which is a string. Transform to int.
    """
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said,
    # the previous day of the first of next month
    return next_month - datetime.timedelta(days=next_month.day)

def create_periods(start_tr: str, pred_month: str):
    """
    Define start/end date for training and prediction sets

    Args:
        start_tr (str): the start date of the training set, used as a filter.
        pred_month (str): the start date of the month that needs to be predicted

    Returns:
        tr_start (str): Start date of the training set.
        tr_end (str): The end date of the training set.
        pred_start (str): Start date of the prediction period.
        pred_end (str): With the end date of the prediction period.

    Exaple:

        .. code-block:: python

            tr_start, tr_end, pred_start, pred_end = createPeriods('2018-01-01', '2022-06-01')

    """
    # TO DO
    # -- ADD AMOUNT OF MONTHS AHEAD THAT NEED TO BE PREDICTED. Using Timedelta?
    # -- MAKE prediction month flexible.
    today = datetime.date.today()
    yesterday = today - timedelta(days=1)
    # call function of last_day_of_month
    last_day_current = int(last_day_of_month(today).day)
    end_current = datetime.date(today.year, today.month, last_day_current)

    start_pred_month = end_current + timedelta(days=1)
    last_day_next = int(last_day_of_month(start_pred_month).day)
    end_next = datetime.date(start_pred_month.year, start_pred_month.month, last_day_next)

    # make start and end dates
    tr_start = start_tr
    tr_end = str(yesterday)
    pred_start = str(start_pred_month)
    pred_end = str(end_next)

    return tr_start, tr_end, pred_start, pred_end

def get_dummies_v2(df: pd.DataFrame, cols: list):
    """
    Function which returns the dummy variables for a given list of columns.
    Args:
        df: pd.DataFrame, an enriched dataframe for a specific country.
        cols: List of columns that need to be transposed to dummy variables.
    Returns:
        df_total: pd.DataFrame, the enriched dataframe where specified columns are changed to dummmy variables.
    Exaple:

        .. code-block:: python

        df = pd.DataFrame([[0,1,2,3,4,5,6], [0,'piet',0,0,'arbeid',0,'jan']]).T
        df.columns = ['weekday', 'Holiday']
        df_with_dummies = get_dummies_v2(df, ['weekday'])
        df_with_dummies
    """
    df_new = df.copy()

    for col in cols:
        dummy = pd.get_dummies(df_new[f'{col}'], prefix=f'{col}')
        print(f'New dummy_cols:\n {list(dummy.columns)} \n {col} is deleted ')
        df_new = df_new.join(dummy)
    df_total = df_new.copy()
    df_total = df_total.drop(columns=cols)

    return df_total

def make_binary(df: pd.DataFrame, cols: list):
    """
    Function which returns the dummy variables for a given list of columns.
    Args:
        df: pd.DataFrame, an enriched dataframe for a specific country.
        cols: List of columns that need to be transposed to binary variables.
    Returns:
        df_total: pd.DataFrame, the enriched dataframe where specified columns are changed to dummmy variables.
    Exaple:

        .. code-block:: python

        df = pd.DataFrame([[0,1,2,3,4,5,6], [0,'piet',0,0,'arbeid',0,'jan']]).T
        df.columns = ['WeekDay', 'Holiday']
        df_with_dummies = make_binary(df, ['Holiday'])
        df_with_dummies

    """
    df_new = df.copy()
    for col in cols:
        col_values = list(df_new[f'{col}'].unique())
        col_values = [str(x) for x in col_values]
        try:
            col_values.remove('0')
        except ValueError:
            pass
        df_new[f'{col}'] = df_new[f'{col}'].replace(col_values, 1)
        df_new[f'{col}'] = df_new[f'{col}'].astype(int)
    return df_new

def add_holiday_col(df: pd.DataFrame, date_col: str, country: str):
    """
    Function to add a column with the national holidays for a specific country.

    Args:
        df (pd.DataFrame): The dataframe where a column with the holidays should be added.
        date_col (str):  The column that contains the date.
        country (str): The country we need to retrieve the holidays for.

    Returns:
        df_new (pd.dataframe): The new dataframe with an additional column that countains the holidays for the specific country.

    """
    df_new = df.copy()
    df_new['Holiday'] = [holidays.country_holidays(f'{country}').get(x) for x in df_new[f'{date_col}']]
    df_new['Holiday'] = df_new['Holiday'].fillna('0')
    return df_new

def create_interval(start: str, end: str, col_name: str, field_val):
    """
    Args:
        start: str, start date for the df with the date as index
        end: str, end date for the df with the date as index
        col_name: str, column name of the date string.
        field_val: placeholder

    Returns:
        df with the index 'date' of the prediction period. and an column with values 0 names interval_name.
    """
    # setting up index date range
    idx = pd.date_range(start, end)
    # create the dataframe using the index above, and creating the empty column for interval_name
    df = pd.DataFrame(index=idx, columns=[f'{col_name}'])
    # set the index name
    df.index.names = ['date']
    # filling out all rows in the 'interval_name' column with the field_val parameter
    df.interval_name = field_val
    return df

def split_data_timewise(df: pd.DataFrame,
                        target: str,
                        tr_start: str,
                        tr_end: str,
                        test_start: str,
                        test_end: str,
                        pred_start: str,
                        pred_end: str,
                        ):
    """
    This function splits the data based on the train/test split given by the parameters in the yaml file
    Args:
        df: pd.DataFrame, an enriched dataframe for a specific country.
        target: str, the target column
        tr_start:
        tr_end:
        test_start: str,
        test_end: str,
        pred_start: str,
        pred_end: str,
        test_size_dataset: float, the value of the training set in decimals (mostly between .05 and .5 )
        random_state: int, the random state, so that we can reproduce values.
    Returns:
        X_train: pd.DataFrame, containing all the feature for the training set.
        X_test: pd.DataFrame, containing all the feature for the test set.
        y_train: pd.DataFrame, containing all real outcomes for the training set.
        y_test: pd.DataFrame, containing all real outcomes for the test set.
    Exaple:

        .. code-block:: python

        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from datetime import datetime
        test = pd.DataFrame(list(np.arange(60)))
        test = pd.DataFrame(test.values.reshape(30,2), columns = ['col1', 'col2'])
        today = datetime.now().strftime("%Y/%m/%d")
        test.index = pd.date_range(start=today, periods=30, freq='D') + pd.Timedelta(days=1)
        tr_start = test.index[5]
        tr_end =  test.index[20]
        X_train, X_test, y_train, y_test = split_data(test, target = 'col1', tr_start = tr_start,
        tr_end = tr_end, test_size_dataset = 0.1)
    """
    data = df.copy()
    X = data.drop(f'{target}', axis=1)
    y = pd.DataFrame(data[f'{target}'])
    tr_end = pd.to_datetime(tr_end) - timedelta(days=1)
    tr_end = str(tr_end)[:10]

    X_train = X.loc[tr_start:tr_end, :].copy()
    y_train = y.loc[tr_start:tr_end, :].copy()
    y_train = y_train[f'{target}']

    X_test = X.loc[test_start:test_end, :].copy()
    y_test = y.loc[test_start:test_end, :].copy()
    y_test = y_test[f'{target}']
    X_predict, y_predict = X.loc[pred_start:pred_end, :].copy(), y.loc[pred_start:pred_end, :].copy()

    return X_train, X_test, y_train, y_test, X_predict, y_predict

def rem_holidays(train: pd.DataFrame, country: str):
    """
    Removes the effect of any holidays (zeros outside weekends) that there might be on the training dataset.

    Args:
        train (pd.DataFrame): Training dataset to be processed
        country (str): International country code
        
    Returns:
        train (pd.DataFrame): Clean training dataset
    """
    holiday = holidays.country_holidays(country.upper())

    

    #Check every row in the training dataset
    for index, i in train.iterrows():

        #If the current index is a holiday
        if index in holiday and index-timedelta(days=7) in train.index:
            #And if the day after is Friday or higher, or if the previous day was a Sunday
            if (index.weekday() > 3) or (index.weekday() == 0):
                #Take same value from last week

                train.at[index, train.columns[0]] = train.at[index - timedelta(days = 7),train.columns[0]]
            #If the current index is in the middle of the week (Tuesday to Thursday)
            else:
                #Take the average of previous and following date
                #Note that in order to take the value for the following date (Date +1) it does not seem possible to do (train.at[index+timedelta(days = 1), train.columns[0]]
                #Instead, we do (train.at[index-timedelta(days = -1), train.columns[0]], which yields the same result without throwing an error
                if index-timedelta(days=-1) in train.index:
                    train.at[index, train.columns[0]] = (train.at[index-timedelta(days = 1), train.columns[0]] + train.at[index-timedelta(days = -1), train.columns[0]])/2
                else:
                    train.at[index, train.columns[0]] = (train.at[index-timedelta(days = 1), train.columns[0]] + train.at[index-timedelta(days = 7), train.columns[0]])/2

    return train

def adj_holidays(train: pd.DataFrame, country: str):
    """
    Removes the effect that any holidays had on historical data. The difference between adj_holidays and rem_holidays
    is that while rem_holidays works with live channels that present a value of 0 during a bank holiday or weekend,
    adj_holidays should be used with non-live channels such as email, for which we get non-zero values every day.

    How it works:
    This function finds each week with a holiday in the training dataset and for each holiday it calculates a prediction
    for its whole week. Then it replaces the value in the day with a bank holiday with the forecasted value.

    Args:
        train (pd.dataframe): Training dataset to be processed
        country (str): International country code

    Returns:
        train (pd.dataframe): Clean train dataset

    Exaple:

        .. code-block:: python

            import pandas as pd
            test_data = {'Daily_volumes':[1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3]}
            date_generated = pd.date_range(start='4/8/2022',end='4/30/2022')
            test_dataframe = pd.DataFrame(test_data, index=date_generated)
            predict_X_plus_5(test_dataframe)

    """
    holiday = holidays.country_holidays(country.upper())

    train.index = pd.DatetimeIndex(train.index.values,freq= 'D')

    #Check every row in the training dataset
    for index, i in train.iterrows():


        #If the current index is a holiday
        if index in holiday and index-timedelta(days=7) in train.index:
            #Get the week's start and end dates
            start = index - timedelta(days=index.weekday())
            
            end = index + timedelta(days= 6 - index.weekday())
            #Forecast that week only and get the forecasted value as the substitution for the holiday
            pred_model = sarimax_fit(train.loc[train.index[0]:start - timedelta(days=1), train.columns[0]])
            
            pred_week = sarimax_predictv2(pred_model, start, end, train.columns[0])

            train.at[index,train.columns[0]] = pred_week.at[index,train.columns[0]]

    return train
################### OUTLIER HANDLING FUNCTIONS ###################

def detect_outliers(column: pd.DataFrame):
        """
        Detects outliers in a single-column dataframe, based on the 1.5*IQR method

        Args:
            column (pd.dataframe): Single-column dataframe to be used

        Returns:
            Single-column dataframe with the collection of outliers found

        """

        if column.empty:
            return column
        
        x = column.to_numpy(dtype=int)

        #Calculate interquartile range
        lqt, hqt = np.quantile(x,[0.25,0.75])

        iqr = hqt - lqt

        #Calculate outlier detection upper bound (we won't bother with the lower bound for now
        #as this specific dataset cannot have negative values and therefore is left skewed)
        upper_bound = hqt + 1.5*iqr

        #Return a dataset with outliers only
        return column.loc[column[column.columns[0]] >= upper_bound]

def detect_outliers_windowed(column: pd.DataFrame, window_size: int = 28):
    """
    Detects outliers in a single-column dataframe, based on the 1.5*IQR method

    Please note that for the windowed version of this function, there will be
    one window that is equal or smaller than the specified window size.

    Args:
        column (pd.DataFrame): Single-column dataframe to be used
        window_size (int): The size (in rows or observations) of the window chosen to perform
        detection on

    Returns:
        outliers_df (pd.DataFrame): Dataframe with the collection of outliers found

    """
    #Delete any 0s
    clean_col = column[(column.T != 0).any()]

    window = clean_col.copy()

    outliers = []

    while len(window) >= window_size:
        window_temp = window.iloc[-window_size:]

        window_outliers = detect_outliers(window_temp)

        outliers.append(window_outliers)

        window.drop(window.index[-window_size:],axis = 0, inplace = True)
    
    window_outliers = detect_outliers(window)

    outliers.append(window_outliers)

    outliers_df = pd.concat(outliers)

    outliers_df.sort_index(ascending=True, inplace=True)

    return outliers_df

def plot_outliers(column: pd.DataFrame, windows_size: int = 28):
    """
    Highlights all outliers detected in a single-column dataframe

    Args:
        column (pd.DataFrame): Single-column dataframe to be used
        window_size (int): The size (in rows or observations) of the window chosen to perform
        detection on

    Returns:
        Prints in terminal a list with all outliers and also renders a plot highlighting them

    """

    out = detect_outliers_windowed(column, window_size=windows_size)

    print(out)

    plt.plot(column)
    plt.plot(out,'ro')
    plt.show()

def remove_outliers(column: pd.DataFrame, window_size: int = 28):
    """
    Removes outliers in a single-column dataframe, based on the 1.5*IQR method

    Args:
        column (pd.DataFrame): Single-column dataframe to be used
        window_size (int): The size (in rows or observations) of the window chosen to perform
        detection on

    Returns:
        treated_df (pd.DataFrame): Clean dataframe with outliers imputed with the window median


    """
    treated_df = column.copy()
    #Delete any 0s
    clean_col = column[(column.T != 0).any()]

    window = clean_col.copy()

    while len(window) > window_size:

        window_temp = window.iloc[-window_size:]

        window_temp.to_clipboard(excel=True)

        window_median = np.median(window_temp)

        window_outliers = detect_outliers(window_temp)
        
        for index, i in window_outliers.iterrows():

            treated_df.loc[index, treated_df.columns[0]] = window_median

        window.drop(window.index[-window_size:],axis = 0, inplace = True)

    window_median = np.median(window)

    window_outliers = detect_outliers(window)
    
    for index, i in window_outliers.iterrows():

        treated_df.loc[index, treated_df.columns[0]] = window_median

    return treated_df

################ END OF OUTLIER HANDLING FUNCTIONS ################

def school_holidays(years: list, country: str):
    """
    Function to create a df with the schoolholidays for the specific country. Taking into account all regions,
    therefore the 'outer boundaries
    Args:
        years: list, list of years you want the holidays from
        country: the country where you need the school holidays from.
    Returns:
        holidays_df: pd.DataFrame, a dataframe with ['Date', 'Holiday'] containing the date and name of the school holiday.
    """
    from workalendar.europe import NetherlandsWithSchoolHolidays as NL
    from vacances_scolaires_france import SchoolHolidayDates

    holidays_list = []
    if country == 'FR':
        regions = ['A','B','C']
    if country == 'NL':
        regions = ['north', 'middle', 'south']

    for region in regions:
        for year in years:
            if country == 'FR':
                try:
                    # Get holiday dates for a given year and zone
                    d = SchoolHolidayDates()
                    df = d.holidays_for_year_and_zone(year, region)
                    df2 = pd.DataFrame.from_dict(df, orient='index')
                    df_holidays = pd.DataFrame(df2.loc[:,['date', 'nom_vacances']])  # print(type(x))
                    # df_holidays = df_holidays.loc[:,['date', 'nom_vacances']]
                    df_holidays.columns = ['Date', 'Holiday']
                    holidays_list.append(df_holidays)
                except BaseException:
                    pass

            elif country == 'NL':
                spring = False
                if region == 'south':
                    spring = True
                calendar = NL(region=region, carnival_instead_of_spring=spring)
                # Get a list of holidays
                holiday_list = calendar.holidays(year)
                # Make a dictionary with a list of holidays for each date entry
                holiday_dict = {}
                for h in holiday_list:
                    holiday_dict.setdefault(h[0], []).append(h[1])
                df_holidays = pd.DataFrame.from_dict(holiday_list)
                holidays_list.append(df_holidays)
            else:
                print(f'Invalid country')
                break
    holidays_df = pd.concat(holidays_list)
    holidays_df.columns = ['Date', 'Holiday']
    holidays_df = holidays_df.drop_duplicates(subset=['Date'], keep='first')
    holidays_df.reset_index(drop=True)

    return holidays_df

### OLD FUNCTIONS

def enrich_dataframe(df: pd.DataFrame, date_splitted: bool):
    """
    Enrichting a dataframe with Date (as datetime), day and month column based on the day, and the week of the day
    (cat 0-6 = mon-sun)
    Args:
        df: pd.DataFrame, an enriched dataframe for a specific country.
        date_splitted: Boolean, to define whether or not you need a date_time column made by the CalendarKey.
    Returns:
        df_new: the new df enriched with columns for Day, Month, Weekday (mon-sun) and Maturity level.
    Exaple:

        .. code-block:: python

        from datetime import datetime
        import pandas as pd
        import numpy as np
        test = pd.DataFrame(list(np.arange(60)))
        test = pd.DataFrame(test.values.reshape(30,2), columns = ['MatureClubs', 'RunningClubs'])
        today = datetime.now().strftime("%Y/%m/%d")
        test.index = pd.date_range(start=today, periods=30, freq='D') + pd.Timedelta(days=1)
        test['Date'] = test.index
        test2 = enrich_dataframe(test, True)
    """

    # TO DO adjust Maturity level.
    df_new = df.copy()
    if date_splitted != True or False:
        return 'Not a Boolean given for date_splitted'
    if date_splitted == True:
        df_new['Date'] = pd.to_datetime(df_new['Date'])
        df_new['Day'] = df_new['Date'].dt.day
        df_new['Month'] = df_new['Date'].dt.month
        # date.weekday() Return the day of the week as an integer, where Monday is 0 and Sunday is 6
        df_new['WeekDay'] = df_new['Date'].dt.weekday.astype(str)

    df_new['MaturityLevel'] = df_new['MatureClubs'] / df_new['RunningClubs']
    df_new['MaturityLevel'] = df_new['MaturityLevel'].replace(np.nan, 0)

    return df_new

def get_dummies(df: pd.DataFrame, cols: list):
    """
    Function which returns the dummy variables for a given list of columns.
    Args:
        df: pd.DataFrame, an enriched dataframe for a specific country.
        cols: List of columns that need to be transposed to dummy variables.
    Returns:
        df_total: pd.DataFrame, the enriched dataframe where specified columns are changed to dummmy variables.
    Exaple:

        .. code-block:: python

        df = pd.DataFrame([[0,1,2,3,4,5,6], [0,'piet',0,0,'arbeid',0,'jan']]).T
        df.columns = ['weekday', 'Holiday']
        df_with_dummies = get_dummies(df, ['weekday'])
        df_with_dummies
    """
    df_new = df.copy()
    if 'Holiday' not in cols:
        holidays_u = list(df_new.Holiday.unique())
        holidays_u = [str(x) for x in holidays_u]
        holidays_u.remove('0')
        df_new.Holiday = df_new.Holiday.replace(holidays_u, 1)

    for col in cols:
        dummy = pd.get_dummies(df_new[f'{col}'], prefix=f'{col}')
        print(f'New dummy_cols:\n {list(dummy.columns)} \n {col} is deleted ')
        df_new = df_new.join(dummy)
    df_total = df_new.copy()
    df_total = df_total.drop(columns=cols)

    return df_total
