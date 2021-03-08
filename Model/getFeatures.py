from dbconnect import dbconnect

def getFeatures():
    """Collect and return the latest value for a list of features
    """
    mydb = dbconnect()

    mycursor = mydb.cursor(buffered=True)
    features = []
    feature_cols = ['AUD_USD',
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

    for col in feature_cols:
        sql = "SELECT feature,price from features WHERE feature = '" + col + "' ORDER BY date DESC LIMIT 1"
        mycursor.execute(sql)
        result = mycursor.fetchall()
        for x in result:
            features.append(x[1])
    return features
