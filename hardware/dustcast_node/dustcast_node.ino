/*
 * DustCast field node -- ESP32 + INA219 + DS18B20 (+ optional pyranometer)
 *
 * WHAT THIS IS FOR
 *
 * Not training data. One node cannot train anything, and the model is already
 * trained on seventeen years of Australian desert PV.
 *
 * Its job is the closed loop. Step 2 of this project established that a
 * one-off conformal calibration under-covers badly once the error distribution
 * drifts -- 76% coverage against a 90% target -- and that recalibrating from a
 * trailing window of OBSERVED error fixes it (90.3%). That fix needs measured
 * output back within about a day. STEG cannot see behind-the-meter rooftop
 * generation, which is the whole problem this project exists to solve, so the
 * only place that feedback can come from is a small monitored sample.
 *
 * This node is one of that sample. It is what makes the calibration layer
 * deployable rather than theoretical -- and, in a demo, it puts a real
 * measured dot inside a band that was predicted before the sun came up.
 *
 * STATUS: written against the datasheets and library APIs, NOT yet run on
 * physical hardware. Treat every calibration constant below as a starting
 * point to be measured, not as a value to trust.
 *
 * WIRING
 *   INA219   I2C    SDA=GPIO21  SCL=GPIO22   (addr 0x40; shunt in the DC line)
 *   DS18B20  1-Wire GPIO4, 4.7k pull-up to 3V3, taped to the panel BACKSHEET
 *   Pyranometer     analog into GPIO34 (ADC1 -- ADC2 is unusable with WiFi on)
 *
 * LIBRARIES  Adafruit_INA219 - OneWire - DallasTemperature - PubSubClient
 *            ArduinoJson
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------- config
static const char* WIFI_SSID     = "CHANGE_ME";
static const char* WIFI_PASSWORD = "CHANGE_ME";
static const char* MQTT_HOST     = "192.168.1.10";
static const uint16_t MQTT_PORT  = 1883;
static const char* MQTT_TOPIC    = "dustcast/telemetry";
static const char* NODE_ID       = "sfax-01";

// Publish cadence. The forecast is hourly, so sub-minute data buys nothing for
// verification -- but a short interval makes the live dot feel alive in a demo
// and costs little on mains power. A battery node should raise this and sleep.
static const uint32_t PUBLISH_INTERVAL_MS = 15000;

// ADC counts -> W/m2 for the pyranometer. MEASURE THIS. The value below is a
// placeholder for a generic 0-1.1 V analogue sensor on the 12-bit ADC at 11 dB
// attenuation. An uncalibrated pyranometer makes the irradiance channel
// decorative, so the node publishes null rather than a guess until it is done.
static const bool  PYRANOMETER_FITTED = false;
static const float PYRANOMETER_W_PER_COUNT = 1600.0f / 4095.0f;

// Nameplate DC rating of the array this node watches, in watts. The backend
// needs it to turn a measurement into a specific yield (kW per kW installed),
// which is the only form in which it can be compared with the forecast.
static const float ARRAY_DC_W = 3000.0f;

static const uint8_t PIN_ONEWIRE = 4;
static const uint8_t PIN_PYRANOMETER = 34;

// ---------------------------------------------------------------- globals
WiFiClient net;
PubSubClient mqtt(net);
Adafruit_INA219 ina219;
OneWire oneWire(PIN_ONEWIRE);
DallasTemperature panelTemp(&oneWire);

bool inaReady = false;
bool tempReady = false;
uint32_t lastPublish = 0;
uint32_t sequence = 0;

// ---------------------------------------------------------------- helpers
void connectWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  // Bounded wait, then return and let loop() retry. Blocking here until
  // connected is how a field node becomes a brick after a router reboot.
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
}

void connectMqtt() {
  if (mqtt.connected()) return;
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  char clientId[48];
  snprintf(clientId, sizeof(clientId), "dustcast-%s-%04X", NODE_ID,
           (uint16_t)(ESP.getEfuseMac() & 0xFFFF));
  mqtt.connect(clientId);
}

float readIrradiance() {
  if (!PYRANOMETER_FITTED) return NAN;
  // Average a burst: the ESP32 ADC is noisy and a single sample wanders by
  // tens of counts, which at this scale is tens of W/m2.
  uint32_t total = 0;
  for (int i = 0; i < 16; i++) { total += analogRead(PIN_PYRANOMETER); delay(2); }
  return (total / 16.0f) * PYRANOMETER_W_PER_COUNT;
}

// ---------------------------------------------------------------- setup
void setup() {
  Serial.begin(115200);
  delay(200);

  Wire.begin(21, 22);
  inaReady = ina219.begin();
  if (inaReady) {
    // The library default is 32V/2A. A 3 kW array runs far above 2 A, so the
    // external shunt must be sized for the real current and this range set to
    // match it -- otherwise every reading silently saturates at full scale.
    ina219.setCalibration_32V_2A();
  }

  panelTemp.begin();
  tempReady = panelTemp.getDeviceCount() > 0;
  panelTemp.setResolution(11);          // ~0.125 degC, ~375 ms conversion

  analogReadResolution(12);
  analogSetPinAttenuation(PIN_PYRANOMETER, ADC_11db);

  connectWifi();
  connectMqtt();
}

// ---------------------------------------------------------------- loop
void loop() {
  connectWifi();
  connectMqtt();
  mqtt.loop();

  if (millis() - lastPublish < PUBLISH_INTERVAL_MS) return;
  lastPublish = millis();

  float volts = NAN, amps = NAN, watts = NAN;
  if (inaReady) {
    // Bus voltage plus the drop across the shunt is the true array voltage.
    // Bus voltage alone under-reads by the shunt drop, which is not negligible
    // at the currents a real string pulls.
    volts = ina219.getBusVoltage_V() + (ina219.getShuntVoltage_mV() / 1000.0f);
    amps  = ina219.getCurrent_mA() / 1000.0f;
    watts = volts * amps;
  }

  float tPanel = NAN;
  if (tempReady) {
    panelTemp.requestTemperatures();
    float t = panelTemp.getTempCByIndex(0);
    if (t > -100.0f) tPanel = t;        // DEVICE_DISCONNECTED_C is -127
  }

  JsonDocument doc;
  doc["node"] = NODE_ID;
  doc["seq"] = sequence++;
  doc["uptime_s"] = millis() / 1000;
  doc["array_dc_w"] = ARRAY_DC_W;
  // Nulls, not zeros, for absent sensors. A zero here is indistinguishable
  // from a panel genuinely producing nothing at night, and the backend would
  // happily average it into a specific yield.
  if (isnan(volts))  doc["voltage_v"] = nullptr; else doc["voltage_v"] = volts;
  if (isnan(amps))   doc["current_a"] = nullptr; else doc["current_a"] = amps;
  if (isnan(watts))  doc["power_w"]   = nullptr; else doc["power_w"]   = watts;
  if (isnan(tPanel)) doc["panel_c"]   = nullptr; else doc["panel_c"]   = tPanel;
  float ghi = readIrradiance();
  if (isnan(ghi))    doc["ghi_wm2"]   = nullptr; else doc["ghi_wm2"]   = ghi;
  doc["rssi"] = WiFi.RSSI();

  char payload[320];
  size_t len = serializeJson(doc, payload, sizeof(payload));

  if (mqtt.connected()) {
    mqtt.publish(MQTT_TOPIC, (const uint8_t*)payload, len, false);
  }
  Serial.println(payload);
}
