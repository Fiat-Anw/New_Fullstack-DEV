"""
CommandCenter File
read from influx and publish to mqtt broker
"""

# Importing relevant modules
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, Dialect
from influxdb_client.client.write_api import ASYNCHRONOUS
import paho.mqtt.client as mqtt
import time
import json
import requests
import pandas as pd

# Load environment variables from ".env"
load_dotenv()

# InfluxDB config
BUCKET = 'fullstack-influxdb' #  bucket is a named location where time series data is stored.
url = 'https://iot-group2-service1.iotcloudserve.net/'
token ='Dwj0HPIYScc1zvkB0zHpjxIVIssU_z_-unniio7sOcZl135FZ40ONj9ZX6jgiBWqkwpOQegRAL21Ix1z86SBJw=='
org = 'Chulalongkorn'
print("Connecting to", url)

client = InfluxDBClient(

    url= url,
    token= token,
    org= 'Chulalongkorn'
)
write_api = client.write_api()
query_api = client.query_api()


csv_result = query_api.query_csv('from(bucket:"fullstack-influxdb") |> range(start: -10m)',
                                 dialect=Dialect(header=False, delimiter=",", comment_prefix="#", annotations=[],
                                                 date_time_format="RFC3339"))


df = pd.DataFrame(csv_result)
print("Successfully convert to DataFrame")
print(df.iloc[:-5])
#df.to_csv('RaspberryPi/CommandCenter/12_5_12_50.csv', index=False)

client.close()
