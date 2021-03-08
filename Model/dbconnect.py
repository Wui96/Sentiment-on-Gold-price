import mysql.connector

def dbconnect(host="localhost",user="root",password="",database="test"):
    """Connect to a mysql database, and return the connection
     Args:
        host ([str]): [The host of mysql]
        user ([str]): [Username for the connection]
        password ([str]): [Password for the user]
        database ([str]): [Name of the database]
    """
    # Connect to db
    mydb = mysql.connector.connect(
        host = host,
        user = user,
        password = password,
        database = database
    )
    
    return mydb