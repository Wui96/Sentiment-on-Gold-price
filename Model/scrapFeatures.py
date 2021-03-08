import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import dateutil.parser
import time
import csv
from datetime import datetime
import io
import sys
from pathlib import Path
import mysql.connector
from dbconnect import dbconnect

def scrapFeatures():
    """Visit the links in list and get the latest value for each features, and save them in database
    """
    _dir = Path(sys.path[0])

    mydb = dbconnect()
    
    mycursor = mydb.cursor(buffered=True)

    driver = webdriver.Chrome(_dir / 'resource' / 'chromedriver.exe') # Initialization

    features = ['AUD_USD',
            'Barrick GOLD',
            'Dow Jones Industrial Average',
            'EGO',
            'Gold Futures',
            'NASDAQ Composite',
            'NCM',
            'NEM',
            'NYSE Composite',
            'NZD_USD',
            'Palladium Futures',
            'Platinum Futures',
            'PLZL',
            'Silver Futures',
            'US Dollar Index Futures',
            'USD_CAD']

    links = [
        'https://www.investing.com/currencies/aud-usd-historical-data',
        'https://www.investing.com/equities/barrick-gold-corp.-historical-data',
        'https://www.investing.com/indices/us-30-historical-data',
        'https://www.investing.com/equities/eldorado-gold-corp.-historical-data?cid=20816',
        'https://www.investing.com/commodities/gold-historical-data',
        'https://www.investing.com/indices/nasdaq-composite-historical-data',
        'https://www.investing.com/equities/newcrest-mining-limited-historical-data',
        'https://www.investing.com/equities/newmont-mining-historical-data',
        'https://www.investing.com/indices/nyse-composite-historical-data',
        'https://www.investing.com/currencies/nzd-usd-historical-data',
        'https://www.investing.com/commodities/palladium-historical-data',
        'https://www.investing.com/commodities/platinum-historical-data',
        'https://www.investing.com/equities/polyus-zoloto_rts-historical-data',
        'https://www.investing.com/commodities/silver-historical-data',
        'https://www.investing.com/currencies/us-dollar-index-historical-data',
        'https://www.investing.com/currencies/usd-cad-historical-data'
    ]

    count = 0

    for link in links:
        # Connect to the links
        driver.get(link)
        
        # Navigate to the values
        table = driver.find_element_by_class_name("historicalTbl")
        tbody = table.find_element_by_tag_name("tbody")
        row = tbody.find_element_by_tag_name("tr")
        date = row.find_elements_by_tag_name("td")[0].text
        price = row.find_elements_by_tag_name("td")[1].text
        
        # Transform and save the values
        date = pd.to_datetime(date,format="%b %d, %Y")
        val = (features[count],str(date.date()),float(price.replace(',' , '')))
        count += 1
        mycursor.execute("INSERT INTO features (feature,date,price) VALUES (%s,%s,%s)",val)
    
    # Commit the SQLs and close the selenium browser
    mydb.commit()
    driver.close()