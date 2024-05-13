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
import pytz
import time
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
MQTT_SUBSCRIBE_TOPIC = "@msg/cc2broker"
print("connecting to MQTT Broker", MQTT_BROKER_URL)
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.connect(MQTT_BROKER_URL,1883)

# read from influx and pub to mqtt
def on_connect(client, userdata, flags, rc, properties):
    """ The callback for when the client connects to the broker."""
    print("Connected with result code "+str(rc))

# def extract_data():
query = 'from(bucket:"fullstack-influxdb")\
|> range(start: -12h)\
|> filter(fn:(r) => r._measurement == "prediction_data")\
|> filter(fn:(r) => r._field == "try8")'

while True:
    csv_result = query_api.query_csv(query,
                                    dialect=Dialect(header=False, delimiter=",", comment_prefix="#", annotations=[],
                                                    date_time_format="RFC3339"))
    df = pd.DataFrame(csv_result)

    columns_to_keep = [5, 6]
    df = df[df.columns[columns_to_keep]]

    new_column_names = ['time','temp_bmp280']
    # Rename the columns
    df = df.rename(columns=dict(zip(df.columns, new_column_names)), inplace=False)

    # Convert 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Convert 'time' column to Thailand timezone
    thailand_tz = pytz.timezone('Asia/Bangkok')
    df['time'] = df['time'].dt.tz_convert(thailand_tz)

    # Format 'time' column as desired
    df['time'] = df['time'].dt.strftime('%Y/%m/%d %H:%M:%S')

    datastr = f"temp_bmp280: {df['temp_bmp280'].iloc[-1]}"
    result = mqttc.publish(MQTT_SUBSCRIBE_TOPIC, datastr)
    status = result[0]
    if status == 0:
        print(f"Send `{datastr}` to topic `{MQTT_SUBSCRIBE_TOPIC}`")
    else:
        print(f"Failed to send message to topic {MQTT_SUBSCRIBE_TOPIC}")

    mqttc.on_connect = on_connect
    time.sleep(6)

