from dbconnect import dbconnect

def getHeadline():
    """Retrieve and combine the headline from the database and return it in a list.
    """
    mydb = dbconnect()
    mycursor = mydb.cursor(buffered=True)
    sql = "SELECT headline from news WHERE datediff(NOW(),date) = 2"
    mycursor.execute(sql)
    myresult = mycursor.fetchall()

    headlines = ""

    for x in myresult:
        headlines = headlines + x[0] + " "
    return [headlines]