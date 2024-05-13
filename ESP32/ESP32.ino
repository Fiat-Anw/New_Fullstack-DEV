#include <WiFi.h>
#include <time.h>
#include <FreeRTOS.h>
#include <task.h>
#include <Adafruit_HTS221.h>
#include <Adafruit_BMP280.h>
#include <PubSubClient.h>


const char *ssid = "--"; // Wifi SSID
const char *password = "--"; // Wifi Password
const char *ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 3600*7;  // Your timezone offset in seconds thailand gmt +7hours = 3600*7 seconds

// Daylight saving time offset in seconds is the practice of advancing clocks (typically by one hour) 
// during warmer months so that darkness falls at a later clock time.

// Thailand currently observes Indochina Time (ICT) all year. Daylight Saving Time has never been used here.
// Clocks do not change in Thailand. There is no previous DST change in Thailand.
const int daylightOffset_sec = 0; // 1 hour = 3600 sec

// Global variables for time
struct tm timeinfo;

// MQTT Broker
const char *mqtt_server = "broker.netpie.io";
const int mqtt_port = 1883;
const char *mqtt_username = "--"; // your netpie device token
const char *mqtt_password = "--"; // your netpie device secret
const char *mqtt_client_id = "--"; // your netpie device client id

#define UPDATEDATA  "@shadow/data/update" //topic for publish

WiFiClient espClient;
PubSubClient client(espClient);

QueueHandle_t sensorDataQueue;

// Function for display Local time
void DisplayLocalTime(void *parameter) {
  for (;;) {

    if (getLocalTime(&timeinfo)) {
      // Print Local time
      Serial.print("Local Time: ");
      Serial.printf("%02d:%02d:%02d\n", timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
    } else {
      Serial.println("Failed to display Time");
    }
    vTaskDelay(pdMS_TO_TICKS(1000)); // delay for 1 second
  }
}

// Function for updating Time with NTP time
void UpdateNTPTime(void *parameter) {
  for (;;) {

    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

    if (getLocalTime(&timeinfo)) {
      Serial.println("Time updated with NTP server");
    } else {
      Serial.println("Failed to update time with NTP server");
    }
    vTaskDelay(pdMS_TO_TICKS(60000)); // Delay for 1 minute
  }
}

// Function to read data from HTS221 and BMP280 sensors
void readSensors(void *parameter) {
  // Initialize HTS221 sensor
  Adafruit_HTS221 hts221;

  // Initialize BMP280 sensor
  Adafruit_BMP280 bmp280;

  // Initialize sensors
  if (!hts221.begin_I2C() || !bmp280.begin()) {
      Serial.println("Error initializing sensors!");
      vTaskDelete(NULL);
  }

  // Serial.println("Reading Sensor Data"); // Debug print

  for (;;) {
    // Read temperature and humidity from HTS221 sensor
    sensors_event_t temp;
    sensors_event_t humid;
    hts221.getEvent(&humid, &temp);// populate temp and humidity objects with fresh data
    float temperature = temp.temperature;
    // Serial.print("Temperature: "); Serial.print(temp.temperature); Serial.println(" degrees C");
    float humidity = humid.relative_humidity;
    // Serial.print("Humidity: "); Serial.print(humidity); Serial.println("% rH");

    // Read temperature and pressure from BMP280 sensor
    float temp_bmp280 = bmp280.readTemperature();
    float pressure_bmp280 = bmp280.readPressure() / 100.0; // Convert Pa to hPa

    // Add sensor data to the queue
    char json_body[200];
    const char json_tmpl[] = "{\"data\":{\"temp_hts221\": %.2f,"
                             "\"humid_hts221\": %.2f," 
                             "\"temp_bmp280\": %.2f," 
                             "\"pressure_bmp280\": %.2f}}";
    sprintf(json_body, json_tmpl, temperature, humidity, temp_bmp280, pressure_bmp280);

    Serial.println(json_body);
        
    // Queue JSON message to the back of the FIFO queue
    xQueueSendToBack(sensorDataQueue, &json_body, portMAX_DELAY);

    // Serial.println("Data pushed to queue"); // Debug print

    // Sleep for 1 minute
    vTaskDelay(pdMS_TO_TICKS(60000));
  }
}

// Function reconnect to wait until connect Netpie Device
void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect(mqtt_client_id, mqtt_username, mqtt_password)) {
      Serial.println("connected");
      // Once connected, subscribe to the topic(s) you wish to receive messages on
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

// Function to publish sensor data to NETPIE
void publishToNETPIE(void *parameter) {
  char json_body[200]; // Buffer to hold JSON message

  for (;;) {
    // Check if the queue has any data
    if (uxQueueMessagesWaiting(sensorDataQueue) > 0) {
      // Receive data from the queue
      if (xQueueReceive(sensorDataQueue, &json_body, portMAX_DELAY) == pdPASS) {
        // Publish data to Netpie
        if (client.connected()) {
          client.publish(UPDATEDATA, json_body);
        } else {
          // If MQTT connection is not established, attempt to reconnect
          reconnect();
        }
      }
    }

    // Delay for a short time before checking the queue again
    vTaskDelay(pdMS_TO_TICKS(10000)); // Publish every 10 seconds
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  // Connect to Wi-Fi
  Serial.print("Connecting to :");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi..."); // Wait for connecting Wifi
  }
  Serial.println("Connected to WiFi"); // Show when Wifi already Connect
  Serial.println("");

  // Create a queue to hold sensor data messages
  sensorDataQueue = xQueueCreate(10, sizeof(char[200])); // queue can hold up to 10 items, each item being a character array of size 200 bytes.

  // Setup MQTT client
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  //create task Name "DisplayLocalTime"
  xTaskCreate(
    DisplayLocalTime,        // Call Function
    "DisplayTimeTask",      // Task name
    10000,                   // Stack size (bytes)
    NULL,                    // Parameter
    0,                       // Task priority
    NULL                    // Task handle
  );

  //create task Name "UpdateNTPTime"
  xTaskCreate(
    UpdateNTPTime,          // Call Function
    "UpdateNTPTimeTask",    // Task name
    10000,                  // Stack size (bytes)
    NULL,                   // Parameter
    1,                      // Task priority (higher priority and DisplayTimeTask)
    NULL                    // Task handle
  );

  //create task Name "readSensors"
  xTaskCreate(
    readSensors,         // Call Function
    "ReadSensorsTask",   // Task name
    10000,               // Stack size (bytes)
    NULL,                // Parameter
    2,                   // Task priority
    NULL                 // Task handle
  );

  // Create task Name "publishToNETPIE"
  xTaskCreate(
    publishToNETPIE,          // Call Function
    "PublishToNETPIETask",   // Task name
    10000,                    // Stack size (bytes)
    NULL,                     // Parameter
    3,                        // Task priority
    NULL                      // Task handle
  );

}

void loop() {
  // reconnect when client disconnect
  if (!client.connected()) {
    reconnect();
  }
  // client.loop();
}