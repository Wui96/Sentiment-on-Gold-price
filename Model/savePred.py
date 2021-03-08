from dbconnect import dbconnect
from datetime import datetime

def savePred(pred):
    """Save the prediction into database
     Args:
        pred ([float]): [The prediction]
    """
    mydb = dbconnect()
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("SELECT date from prediction")
    result = mycursor.fetchone()
    if result:
        return 0
    else:
        mycursor.execute("INSERT INTO prediction (date,pred) VALUES (%s,%s)",[datetime.today().strftime('%Y-%m-%d'),str(pred)])
        mydb.commit()