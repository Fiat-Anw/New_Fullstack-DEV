#include <WiFi.h>
#include <Wire.h>
#include "PubSubClient.h"
#include <time.h>
#include "Adafruit_SHT4x.h"
#include <Adafruit_BMP280.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

//for netpie
//#define UPDATEDATA   "@shadow/data/update"

#define UPDATEDATA   "@msg/sensor2broker"
// WiFi Connection     
const char* ssid      = "ultra_wifi";      
const char* password  = "12345678";     

const char *ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 3600*7;  // Your timezone offset in seconds thailand gmt +7hours = 3600*7 seconds

// Daylight saving time offset in seconds is the practice of advancing clocks (typically by one hour) 
//during warmer months so that darkness falls at a later clock time.

//Thailand currently observes Indochina Time (ICT) all year. Daylight Saving Time has never been used here.
// Clocks do not change in Thailand. There is no previous DST change in Thailand.
const int daylightOffset_sec = 0; // 1 hour = 3600 sec

// Global variables for time
struct tm timeinfo;

const char* mqtt_server = "172.20.10.3";
const int mqtt_port = 1883;
const char* token = "NULL";
const char* secret = "NULL";
const char* client_id = "ESP32GROUP2";

Adafruit_SHT4x sht4x = Adafruit_SHT4x();
Adafruit_BMP280 bmp280;
WiFiClient espClient;
PubSubClient client(espClient);

QueueHandle_t sensorDataQueue;


void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print(F("Connecting to "));
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(F("."));
  }
  Serial.println(F(""));
  Serial.println(F("WiFi connected"));
  Serial.println(F("IP address: "));
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.println(F("Attempting MQTT connection..."));
    //if (client.connect(client_id, token, secret)) {
    if (client.connect("ESP32GROUP2")) {
      Serial.println(F("connected"));
      // Subscribe
      client.subscribe("@msg/cc2broker");
    } else {
      Serial.print(F("failed, rc="));
      Serial.print(client.state());
      Serial.println(F(" try again in 5 seconds"));
      delay(2000);
    }
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print(F("Message arrived ["));
  Serial.print(topic);
  Serial.print(F("] "));
  // for (int i = 0; i < length; i++) {
  for (int i = 0; i < 18; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
}

void displayLocalTime(void *parameter) {
  for (;;) {

    if (getLocalTime(&timeinfo)) {
      // Serial.print("Local Time: ");
      // Serial.printf("%02d:%02d:%02d\n", timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
      Serial.print("");
    } else {
      Serial.println("Failed to obtain time");
    }
    vTaskDelay(pdMS_TO_TICKS(1000)); // delay for 1 second
  }
}

void updateNTPTime(void *parameter) {
  for (;;) {

    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

    if (getLocalTime(&timeinfo)) {
      //Serial.println("Time updated with NTP server");
    } else {
      Serial.println("Failed to update time with NTP server");
    }
    vTaskDelay(pdMS_TO_TICKS(60000)); // Delay for 1 minute
  }
}

void sendSensorData(void *parameter) {
  for (;;) {

    sensors_event_t humidity, temp;
    sht4x.getEvent(&humidity, &temp); // Populate temp and humidity objects with fresh data
    
    float temp_sht4x = temp.temperature;
    float humid_sht4x = humidity.relative_humidity;
      
    float temp_bmp280 = bmp280.readTemperature();
    float pressure_bmp280 = bmp280.readPressure() / 100.0F; // Convert Pa to hPa

    char json_body[200];
    /*
    for netpie
    const char json_tmpl[] = "{\"data\":{\"temp_sht4x\": %.2f,"
                             "\"humid_sht4x\": %.2f," 
                             "\"temp_bmp280\": %.2f," 
                             "\"pressure_bmp280\": %.2f}}";
    */

    // for Proj4
    const char json_tmpl[] = "{\"temp_sht4x\": %.2f,"
                             "\"humid_sht4x\": %.2f," 
                             "\"temp_bmp280\": %.2f," 
                             "\"pressure_bmp280\": %.2f}";

    sprintf(json_body, json_tmpl, temp_sht4x, humid_sht4x, temp_bmp280, pressure_bmp280);

    //Serial.println(json_body);

    Serial.print(F("Read_SensorData ["));
    Serial.print("ReadTempBMP280");
    Serial.print(F("] temp_bmp280: "));
    Serial.print(temp_bmp280);
    Serial.println();
        
    // Queue JSON message to the back of the FIFO queue
    xQueueSendToBack(sensorDataQueue, &json_body, portMAX_DELAY);

    vTaskDelay(pdMS_TO_TICKS(6000)); // Read sensor data every 6 seconds
  }
}

void publishDataFromQueue(void *parameter) {
  char json_body[200]; // Buffer to hold JSON message

  for (;;) {
    // Check if the queue has any data
    if (uxQueueMessagesWaiting(sensorDataQueue) > 0) {
      // Receive data from the queue
      if (xQueueReceive(sensorDataQueue, &json_body, portMAX_DELAY) == pdPASS) {
        // Publish data to Netpie
        if (client.connected()) {
          client.publish(UPDATEDATA, json_body);
          //Serial.println("pubDataFromQueue");
        } else {
          // If MQTT connection is not established, attempt to reconnect
          reconnect();
        }
      }
    }

    // Delay for a short time before checking the queue again
    vTaskDelay(pdMS_TO_TICKS(6000)); // Publish every 6 seconds
  }
}


void setup() {
  Serial.begin(115200);
  if (!sht4x.begin()) {
    Serial.println("Couldn't find SHT4x");
    while (1) delay(1);
  }
  if (!bmp280.begin()) {
    Serial.println("Failed to initialize BMP280 sensor!");
    while (1);
  }

  // Create a queue to hold sensor data messages
  sensorDataQueue = xQueueCreate(10, sizeof(char[200])); // queue can hold up to 10 items, each item being a character array of size 200 bytes.

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  // Subscribe
  //client.subscribe("@msg/postman");

  client.setCallback(callback);

  xTaskCreate(
    displayLocalTime,        // Function to run
    "DisplayTimeTask",      // Task name
    10000,                   // Stack size (bytes)
    NULL,                    // Parameter to pass
    1,                       // Task priority
    NULL                    // Task handle
  );

  xTaskCreate(
    updateNTPTime,          // Function to run
    "UpdateNTPTimeTask",    // Task name
    10000,                   // Stack size (bytes)
    NULL,                    // Parameter to pass
    2,                       // Task priority (higher priority)
    NULL                    // Task handle
  );

  xTaskCreate(
    sendSensorData,     // Task function
    "sendSensorData",   // Task name
    10000,              // Stack size (bytes)
    NULL,               // Parameter to pass to the task
    3,                  // Task priority
    NULL                // Task handle
  );
  xTaskCreate(
    publishDataFromQueue,     // Task function
    "publishDataFromQueue",   // Task name
    10000,                    // Stack size (bytes)
    NULL,                     // Parameter to pass to the task
    4,                        // Task priority
    NULL                      // Task handle
  );
}

void loop() {
  // Empty because all operations are handled in FreeRTOS tasks

  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
