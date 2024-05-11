"""
CommandCenter File
read from influx and publish to mqtt broker
"""

# Importing relevant modules
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS
import paho.mqtt.client as mqtt
import time
import json
import requests

# Load environment variables from ".env"
load_dotenv()

# InfluxDB config
BUCKET = os.environ.get('INFLUXDB_BUCKET') #  bucket is a named location where time series data is stored.
print("connecting to",os.environ.get('INFLUXDB_URL'))
client = InfluxDBClient(

    url=str(os.environ.get('INFLUXDB_URL')),
    token=str(os.environ.get('INFLUXDB_TOKEN')),
    org=os.environ.get('INFLUXDB_ORG')
)
write_api = client.write_api()
query_api = client.query_api()
 
# MQTT broker config
MQTT_BROKER_URL = os.environ.get('MQTT_URL')
MQTT_SUBSCRIBE_TOPIC = "@msg/influx2broker"
print("connecting to MQTT Broker", MQTT_BROKER_URL)
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.connect(MQTT_BROKER_URL,1883)

def on_connect(client, userdata, flags, rc, properties):
    """ The callback for when the client connects to the broker."""
    print("Connected with result code "+str(rc))

# read from influx and publish to mqtt broker
def read_from_influxdb():
    tables = query_api.query('from(bucket:BUCKET) |> range(start: -10m)')
    for table in tables:
        print(table)
        for record in table.records:
            print(record.values)
            msg = f"messages: {record.values}"
            result = mqttc.publish(MQTT_SUBSCRIBE_TOPIC, msg)
            status = result[0]
            if status == 0:
                print(f"Send `{msg}` to topic `{MQTT_SUBSCRIBE_TOPIC}`")
            else:
                print(f"Failed to send message to topic {MQTT_SUBSCRIBE_TOPIC}")




mqttc.on_connect = on_connect
mqttc.loop_forever()
