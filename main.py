"""
Program: Customer Service Forecasting System. The collected data is processed and analyzed using the
seasonal ARIMA algorithm to generate a forecast for up to 5 weeks in the future.

Description: This script collects historical data and generates a forecast for a given time period.
It uses various data sources to gather information, such as Salesforce or the HDC inlcuding missed
chats from Seamly be customized based on user preferences and input.

Author: Leonardo Jul Camargo

Department: Basic-Fit Customer Service Workforce Management

Date: 2023-03-01

Version: 1.1
"""
#Import custom libraries
import jobs as job

#Import IO and connectivity libraries
import os
import requests
import traceback

#Declare text formatting class
class bcolors:
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Gather all historical data and produce a forecast for each channel
try:
    job.all_cols_x_plus_x()

    print(fr"""{bcolors.BLUE}{bcolors.BOLD}
 ________  ___  ___  ________  ________  _______   ________   ________  ___  ___       
|\   ____\|\  \|\  \|\   ____\|\   ____\|\  ___ \ |\   ____\ |\   ____\|\  \|\  \      
\ \  \___|\ \  \\\  \ \  \___|\ \  \___|\ \   __/|\ \  \___|_\ \  \___|\ \  \ \  \     
 \ \_____  \ \  \\\  \ \  \    \ \  \    \ \  \_|/_\ \_____  \\ \_____  \ \  \ \  \    
  \|____|\  \ \  \\\  \ \  \____\ \  \____\ \  \_|\ \|____|\  \\|____|\  \ \__\ \__\   
    ____\_\  \ \_______\ \_______\ \_______\ \_______\____\_\  \ ____\_\  \|__|\|__|   
   |\_________\|_______|\|_______|\|_______|\|_______|\_________\\_________\  ___  ___ 
   \|_________|                                      \|_________\|_________| |\__\|\__\          
                                                                             \|__|\|__|          
                                                            
{bcolors.ENDC}""")

except:

    print(f"{bcolors.RED}{bcolors.BOLD}Job did not finish, see stack{bcolors.ENDC}")

    traceback.print_exc()