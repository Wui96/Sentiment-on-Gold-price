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
from pathlib import Path
import sys
import mysql.connector
from dbconnect import dbconnect

def scrapNews():
    """Loop through first few page in Reuter's Commodity News and collect the news
    """
    
    # Connect to database
    mydb = dbconnect()
    mycursor = mydb.cursor(buffered=True)

    _dir = Path(sys.path[0])
    driver = webdriver.Chrome(_dir / 'resource' / 'chromedriver.exe') # Initialization

    data = pd.DataFrame(columns=['date','headline','link']) # New dataframe
    data.iloc[0:0]
    
    # Start scrapping from first page with latest news
    driver.get('https://www.reuters.com/news/archive/GCA-Commodities?page=1')
    for x in range(1,5):
        try:
            driver.get('https://www.reuters.com/news/archive/GCA-Commodities?page=' + str(x+1))
            # Wait for a while
            time.sleep(3)
            stories = driver.find_elements_by_class_name("story")
            for i in range(10):
                # Collect information of the news
                headline = stories[i].find_element_by_class_name("story-title").text
                link = stories[i].find_element_by_tag_name("a").get_attribute("href")
                date = stories[i].find_element_by_class_name("timestamp").text
                # Save the result in a DataFrame
                temp = pd.DataFrame({'date':date,'headline':headline,'link':link},index=[x])
                data = data.append(temp)
            print("Date : " + date + " page : " + str(x))
        except Exception as e:
            print(e)           
            break
            
    # Preprocess the data to avoid SQL errors
    data.reset_index(drop=True,inplace=True)
    data['date'] = pd.to_datetime(data['date'],format='%b %d %Y').dt.date
    data['headline'] = data['headline'].apply(lambda x: x.replace('\'',''))
    data['headline'] = data['headline'].apply(lambda x: x.replace('\"',''))

    # Check if there is existing record in the database, and insert if no
    row = 0
    for index,rows in data.iterrows():
        sql = "SELECT link from news WHERE link = '" + rows['link'] +"'"
        mycursor.execute(sql)
        result = mycursor.fetchone()
        if result:
            break
        sql = "INSERT INTO news (date, headline, link) VALUES (\"" + str(data['date'][row]) + "\" , \"" + data['headline'][row] + "\" , \"" + data['link'][row] + "\")"
        mycursor.execute(sql)
        row += 1

    mydb.commit()
    driver.close()