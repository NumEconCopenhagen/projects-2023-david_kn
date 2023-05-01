import datetime

#Function converting specific string ('xxxxMxx') to datetime component:

def date_conv(x):

    # define the input string
    input_string = x['Time']

    # parse the year and month from the input string
    year = int(input_string[:4])
    month = int(input_string[5:])

    # create a datetime object using the year and month
    time_component = datetime.datetime(year=year, month=month, day=1)

    return time_component

def DigitRemoveFromCategory(df, old, new):

    for i,x in enumerate(old):

        I = (df.Category == x)

        df.loc[I, ['Category']] = [new[i]]