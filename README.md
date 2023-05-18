## Forecasting System for a Customer Service contact centre

### Background

As a workforce manager it is crucial to know in advance what the staffing demands will be so that these can be efficiently met. An accurate estimation of the workforce required will help us schedule the right amount of customer service agent hours so that all channels will be covered and all department KPIs will be met. In order to know how many agents will be needed to conver each channel on a given day we need an accurate estimation of the incoming work. That is what the forecasting system does. In this document I will present how I designed and implemented the forecasting system, as well as its performance, metrics and reporting counterpart in Power BI.

**INDEX**
1. [How does it work?](#how-does-it-work)
  - [Pre-processing](#pre-processing)
2. [Ticket bottleneck analysis](#ticket-bottleneck-analysis)
3. [Ticket categorization](#ticket-categorization)
4. [Conclussions and recommendations](#conclussions-and-recommendations)


### How does it work?

The Forecasting System is a series of processes and subprocesses that:
1. Gets the data from different places
2. Processes this data to remove any artifacts or effects we do not want to be carried over to the forecast
3. Produces a forecast for a given timeframe
4. Processes this forecast to add any effects known in advance, such as holidays effects or small corrections
5. Saves the forecast where it can be retrieved in the future by other programs or processes

The following flowchart summarizes the building blocks of the system:

<img src ="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/FC_System.png" width="50%" height="50%">

### Pre-processing

Before using the historical data to produce any forecasts we need to make sure that all the data present in the time-series is representative of the distribution we are working with and, therefore, we can extrapolate it to the future. For that we will remove any outliers. Considering that we are working with a poisson distributio, we will divide these outliers into two categories:

1. Values significantly higher than the baseline
2. Values abnormally lower than the baseline

For case 1 we will use:

![](https://latex.codecogs.com/svg.image?x_%7Bi%7D%20%3E%201.5%5Ctimes%20IQR)

For this we just need to calculate the IQR (interquartile range) of our historic data, and compare each reading to this value. All those readings that are **higher** than this will be considered outliers. All those values will be replaced by the median of the distribution. Here is the code to achieve it:

``` python
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
```

For case 2 we use a different method. Since what we want to remove are the effects caused by bank holidays (lower contacts than usual) we forecast what would have been a "normal" day and then use that forecast to substitute the abnormally low value:

<img src="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/holiday-graph-example.jpg" width="50%" height="50%">

As we can see, on the third Wednesday of this time series we got an abnormally low number of contacts. Since we know this was a bank holiday, we can "forecast the past" to find out what would have happened should there had not been a bank holiday on that Wednesday. We will then use the "forecasted" value to imputate the data for a more realistic time-series and avoide carrying over this effect into our forecast:

<img src="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/holiday-graph-corrected.jpg" width="50%" height="50%">


