# Importing relevant modules
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
import paho.mqtt.client as mqtt
import json

# Load environment variables from ".env"
load_dotenv()

# InfluxDB config
INFLUXDB_URL = os.environ.get('INFLUXDB_URL')
INFLUXDB_TOKEN = os.environ.get('INFLUXDB_TOKEN')
INFLUXDB_ORG = os.environ.get('INFLUXDB_ORG')
BUCKET = os.environ.get('INFLUXDB_BUCKET')

# MQTT broker config
MQTT_BROKER_URL = os.environ.get('MQTT_URL')
MQTT_PUBLISH_TOPIC = "@msg/data"

# Connect to InfluxDB
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

# Connect to MQTT broker
mqttc = mqtt.Client()
mqttc.connect(MQTT_BROKER_URL, 1883)

# Subscribe to InfluxDB for data
query = f'from(bucket: "{BUCKET}") |> range(start: -1m)'
tables = client.query_api().query(query, org=INFLUXDB_ORG)

def on_connect(client, userdata, flags, rc):
    """ The callback for when the client connects to the broker."""
    print("Connected to MQTT Broker with result code "+str(rc))

def on_message(client, userdata, msg):
    """ The callback for when a PUBLISH message is received from the server."""
    print(msg.topic+" "+str(msg.payload))

    # Extract data from InfluxDB query result
    data = []
    for table in tables:
        for row in table.records:
            data.append(row.values)
    
    # Publish data to MQTT broker
    mqttc.publish(MQTT_PUBLISH_TOPIC, json.dumps(data))

# Register MQTT callbacks
mqttc.on_connect = on_connect
mqttc.on_message = on_message

# Start MQTT client loop
mqttc.loop_forever()