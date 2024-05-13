"""
CommandCenter File
read from influx and publish to mqtt broker
"""

# Importing relevant modules
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point,  Dialect
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

# MQTT broker config
MQTT_BROKER_URL = "172.20.10.3"
MQTT_SUBSCRIBE_TOPIC = "@msg/datacc2broker"
print("connecting to MQTT Broker", MQTT_BROKER_URL)
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.connect(MQTT_BROKER_URL,1883)

# read from influx and pub to mqtt
def on_connect(client, userdata, flags, rc, properties):
    """ The callback for when the client connects to the broker."""
    print("Connected with result code "+str(rc))

def extract_data():
    query = 'from(bucket:"fullstack-influxdb")\
    |> range(start: -3d)\
    |> filter(fn:(r) => r._measurement == "sensor_data")\
    |> filter(fn:(r) => r._field == "predict")'

    query2 = 'from(bucket: "fullstack-influxdb")\
        |> range(start: -12h) \
        |> filter(fn: (r) => r._measurement == "sensor_data")\
        |> filter(fn: (r) => r._field == "humid_sht4x" or r._field == "pressure_bmp280" or r._field == "temp_bmp280" or r._field == "temp_sht4x")\
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'


    # csv_result2 = query_api.query_csv(query2,
    #                                 dialect=Dialect(header=False, delimiter=",", comment_prefix="#", annotations=[],
    #                                                 date_time_format="RFC3339"))
    csv_result = query_api.query_csv(query,
                                    dialect=Dialect(header=False, delimiter=",", comment_prefix="#", annotations=[],
                                                    date_time_format="RFC3339"))
    df = pd.DataFrame(csv_result)
    print(csv_result)

    for index, value in df['predict'].iteritems():
        # publish the data to MQTT Broker
        datastr = f"Time: {index}, Value: {value}"
        result = mqttc.publish(MQTT_SUBSCRIBE_TOPIC, datastr)
        status = result[0]
        if status == 0:
            print(f"Send `{datastr}` to topic `{MQTT_SUBSCRIBE_TOPIC}`")
        else:
            print(f"Failed to send message to topic {MQTT_SUBSCRIBE_TOPIC}")
        

        # mqttc.publish(MQTT_PUBLISH_TOPIC, json.dumps(payload))
        # print(f"Published new measurement: {json.dumps(payload)}")

    # #csv_result =  query_api.query_csv(query2)
    # df = pd.DataFrame(csv_result)
    # print("Successfully convert to DataFrame")
    # # print(df.iloc[:-5])
    # #df.to_csv('RaspberryPi/CommandCenter/12_5_12_50.csv', index=False)

    # client.close()

    # columns_to_keep = [5, 7, 8, 9, 10]
    # df = df[df.columns[columns_to_keep]]

    # new_column_names = ['time', 'humid_sht4x', 'pressure_bmp280', 'temp_bmp280', 'temp_sht4x']

    # df = df.rename(columns=dict(zip(df.columns, new_column_names)), inplace=False)

    # df['time'] = pd.to_datetime(df['time'])
    # # Format 'time' column as desired
    # df['time'] = df['time'].dt.strftime('%Y/%m/%d %H:%M:%S')
    # df.head()


mqttc.on_connect = on_connect
mqttc.loop_forever()

# # Main loop
# while True:
#     extract_data()
#     time.sleep(10)  # Sleep for 10 seconds before the next extraction
