import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from env import host, username, password

#helper functions
import utils


##### DATA ACQUISITION ######
### get_db_url(db_name)
### getNewZillowData()
### getZillowData()

#Function to create database url.  Requires local env.py with host, username and password. 
# No function help text provided as we don't want the user to access it and display their password on the screen
def get_db_url(db_name,user=username,password=password,host=host):
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url

#Function to get new data from Codeup server
def getNewZillowData():
    """
    Retrieves zillow dataset from Codeup DB and stores a local csv file
    Returns: Pandas dataframe
    """
    db_name= 'zillow'
    filename='zillow.csv'
    sql = """
    SELECT *
    FROM properties_2017
        JOIN predictions_2017 USING(parcelid)
        LEFT JOIN airconditioningtype USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE transactiondate LIKE '2017%%' 
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL;
    """
    #Read SQL from file
    df = pd.read_sql(sql,get_db_url(db_name))
    #Drop ID columns
    df.drop(columns=['id'],inplace=True)
    #write to disk - writes index as col 0:
    df.to_csv(filename)
    return df

#Function to get data from local file or Codeup server 
def getZillowData():
    """
    Retrieves Zillow dataset from working directory or Codeup DB. Stores a local copy if one did not exist.
    Returns: Pandas dataframe of zillow data
    """
    #Set filename
    filename = 'zillow.csv'

    if os.path.isfile(filename): #check if file exists in WD
        #grab data, set first column as index
        return pd.read_csv(filename,index_col=[0])
    else: #Get data from SQL db
        df = getNewZillowData()
    return df

##################################
##################################

#############################
##### DATA PREPARATION ######

def only_single_unit(df):
    '''
    This function takes in a zillow dataframe and drops any rows with unitcnt > 2(keeps nulls).
    It also keeps parcels of the following property land use type:  
        Single Family (261), PUD (269), Mobile home (263), Townhouse (264), Condominium (266),
        Manufactured, etc (275), Residential general (260), Rural residence (262), Bungalow (273),
        Zero Lot Line (274), Inferred Single Family (279), Patio Home (276)
    '''
    #only keep rows with unit count of 1 or null
    df = df[(df.unitcnt ==1) | df.unitcnt.isna()]
    #filter land use type
    df = df[df.propertylandusetypeid.isin([261,269,263,264,266,275,260,262,273,274,279,276])]
    return df

def map_encode_zillow_fips(df):
    #map to county names
    df['county'] = df.fips.map({6037: 'LosAngeles',6059:'Orange',6111:'Ventura'})
    #encode into dummy df
    d_df = pd.get_dummies(df['county'],drop_first=True)
    #concat dummy df to the rest
    df = pd.concat([df,d_df],axis=1)
    #Drop fips
    df.drop(columns=['fips'],inplace=True)
    return df

def prep_zillow(df):
    '''
    This function takes in a zillow dataframe and prepares it for exploratory analysis.
    '''
    #FILTER down to 'single unit' properties
    df = only_single_unit(df)
    
    #NULLS: filter down null-heavy columns, then drop rows with nulls
    df = handle_missing_values(df,prop_req_col=.98,prop_req_row=1)
    
    #DROP remaining columns I don't want - do geographic columns separately to easily change
    drp_cols = ['propertylandusetypeid', 'finishedsquarefeet12', 'structuretaxvaluedollarcnt',\
                'assessmentyear','landtaxvaluedollarcnt', 'taxamount']
    drp_geo_cols = ['regionidcity','regionidcounty', 'regionidzip','rawcensustractandblock','censustractandblock']
    df.drop(columns=(drp_cols+drp_geo_cols), inplace=True)
#     df.drop(columns=drp_geo_cols, inplace=True)

    #CONVERT datatypes
    to_int_cols = ['bedroomcnt', 'yearbuilt', 'fullbathcnt', 'latitude', 'longitude','roomcnt','calculatedfinishedsquarefeet']
    df[to_int_cols] = df[to_int_cols].astype(int)
    
    #CONVERT transactiondate
    df['trans_month'] = pd.DatetimeIndex(df['transactiondate']).month
    df.drop(columns='transactiondate',inplace=True)
    
    #MOVE parcelid to index
    df.set_index('parcelid',drop=True,inplace=True)
    
    #MAP and ENCODE fips
    df = map_encode_zillow_fips(df)
    
    #TRIM IQR outliers
    iqr_trim_cols = ['bedroomcnt','calculatedbathnbr','calculatedfinishedsquarefeet', 'taxvaluedollarcnt','logerror']
    df = utils.handle_iqr_outliers(df,trim=True,include=iqr_trim_cols)
    
    #RENAME columns
    rename_cols = {
        'bedroomcnt':'bed',
        'bathroomcnt':'bath',
        'calculatedbathnbr':'calcbath',
        'calculatedfinishedsquarefeet':'sf',
        'latitude': 'lat',
        'longitude':'lon',
        'propertycountylandusecode':'countylanduse',
        'propertylandusedesc':'landusedesc',
        'taxvaluedollarcnt':'value'
    }
    df.rename(columns=rename_cols,inplace=True)
    
    return df    
    
##################################
##################################