from dbconnect import dbconnect
from datetime import datetime

def checkUpdate():
    """Check if there is record of prediction for the day, and return a boolean of record found or not.
    """
    mydb = dbconnect()
    mycursor = mydb.cursor(buffered=True)
    
    mycursor.execute("SELECT pred FROM prediction WHERE date = '" + datetime.today().strftime('%Y-%m-%d') + "'")
    result = mycursor.fetchone()
    
    # Check if prediction is done
    if result:
        return True
    else:
        return False