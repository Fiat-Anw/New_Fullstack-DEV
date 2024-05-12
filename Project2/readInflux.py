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
print("connecting to",url)

client = InfluxDBClient(

    url= url,
    token= token,
    org= 'Chulalongkorn'
)
write_api = client.write_api()
query_api = client.query_api()

#"temp_sht4x": 34.64,"humid_sht4x": 36.80,"temp_bmp280": 35.90,"pressure_bmp280": 1006.34}


query1 = 'from(bucket: "fullstack-influxdb")\
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\
  |> filter(fn: (r) => r["_measurement"] == "sensor_data")\
  |> filter(fn: (r) => r["_field"] == "humid_sht4x" or r["_field"] == "pressure_bmp280" or r["_field"] == "temp_bmp280" or r["_field"] == "temp_sht4x")\
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)\
  |> yield(name: "mean")'

query = 'from(bucket: "fullstack-influxdb")\
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\
  |> filter(fn: (r) => r._measurement == "sensor_data")\
  |> filter(fn: (r) => r._field == "humid_sht4x" or r._field == "pressure_bmp280" or r._field == "temp_bmp280" or r._field == "temp_sht4x")\
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)\
  |> yield(name: "mean")'

query2 = 'from(bucket: "fullstack-influxdb")\
    |> range(start: -1h) \
    |> filter(fn: (r) => r._measurement == "sensor_data")\
    |> filter(fn: (r) => r._field == "humid_sht4x" or r._field == "pressure_bmp280" or r._field == "temp_bmp280" or r._field == "temp_sht4x")\
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'

# test for table
# result = query_api.query(org=org, query=query2)
# print(result)
# results = []
# for table in result:
#   for record in table.records:
#     print(record)
#     break





csv_result = query_api.query_csv(query2,
                                 dialect=Dialect(header=False, delimiter=",", comment_prefix="#", annotations=[],
                                                 date_time_format="RFC3339"))


#csv_result =  query_api.query_csv(query2)
df = pd.DataFrame(csv_result)
print("Successfully convert to DataFrame")
print(df.iloc[:-5])
#df.to_csv('RaspberryPi/CommandCenter/12_5_12_50.csv', index=False)

client.close()
