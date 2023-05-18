## Forecasting System for a Customer Service contact centre

### Background

As a workforce manager it is crucial to know in advance what the staffing demands will be so that these can be efficiently met. An accurate estimation of the workforce required will help us schedule the right amount of customer service agent hours so that all channels will be covered and all department KPIs will be met. In order to know how many agents will be needed to conver each channel on a given day we need an accurate estimation of the incoming work. That is what the forecasting system does. In this document I will present how I designed and implemented the forecasting system, as well as its performance, metrics and reporting counterpart in Power BI.

**INDEX**
1. [How does it work?](#how-does-it-work?)
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

![Flowchart](https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/FC-System.svg)
