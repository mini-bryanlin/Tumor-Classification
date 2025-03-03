import mysql.connector
import pandas as pd
import os
import csv
from sqlalchemy import create_engine
import numpy as np

# Expand the tilde to your actual home directory path
file_path = os.path.expanduser("~/password/sql.txt")

def get_header(file_path):
    data = pd.read_csv(file_path)
    sample = np.array(data)[0]
    column_headers = list(data.columns.values)
    hashmap = {}
    for head in range(len(column_headers)):
        hashmap[column_headers[head]] = str(type(sample[head]))
    print(hashmap)
    return hashmap


file = open(file_path,'r')
password = file.readline()
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = password,
    database = "neuralnetwork"

)
bc_file = os.path.expanduser("~/Tumor-Classification/breast-cancer.csv")
headers = get_header("~/Tumor-Classification/breast-cancer.csv")
mycursor = mydb.cursor()
# mycursor.execute("DROP TABLE BC_DATA")
def load_csv_to_mysql(csv_file_path, table_name, if_exists='replace', 
                     host='localhost', user='root', password='', database=''):
   
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Create connection string
    connection_str = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
    
    # Create SQLAlchemy engine
    engine = create_engine(connection_str)
    
    # Load the data into the table
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    
    print(f"Data from {csv_file_path} successfully loaded to {table_name} table in MySQL database {database}")
# load_csv_to_mysql(file_path,"BC_DATA",password = password, database= "neuralnetwork")

# mycursor.execute("drop table BC_DATA")
# mycursor.execute("create table BC_DATA (id int(255))")
# for header in headers:
#     if header != "id":
#         mycursor.execute(f"alter table BC_DATA add {header} FLOAT ")
