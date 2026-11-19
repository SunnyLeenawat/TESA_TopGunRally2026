#include <Arduino.h>
#include <math.h>
#include "driver/gpio.h"

namespace Method1Calibration {
// constexpr double SLOPE = 3.023923;
// constexpr double INTERCEPT = -16.011730;
constexpr double SLOPE = 3.031309;
constexpr double INTERCEPT = -16.087502;
static_assert(SLOPE > 0.0, "Calibration slope must be positive");

inline double capacitancePf(double medianUs) {
  return SLOPE * medianUs + INTERCEPT;
}
}
constexpr char TEAM_ID[] = "TGR-55";
constexpr int CHARGE_PIN = 25;
constexpr int COMPARATOR_PIN = 27;
constexpr int REFERENCE_PIN = 34;
constexpr int BUTTON_PIN = 26;
constexpr int STATUS_LED_PIN = 2;
constexpr int LED_ON = HIGH;
constexpr int LED_OFF = LOW;
constexpr unsigned SAMPLE_COUNT = 64;
constexpr uint32_t DEBOUNCE_MS = 30;
constexpr uint32_t LED_MS = 200;

volatile bool crossed = false;
volatile uint32_t crossingCycles = 0;
uint32_t elapsedCycles[SAMPLE_COUNT];
bool clockReady = false;
bool calibrationLogging = false;
uint32_t calibrationRun = 0;
uint32_t measuredReferenceMv = 0;
unsigned measuredCpuMhz = 0;
double measuredMedianUs = NAN;
const char *timingStatus = "not_measured";
bool armed = false;
int previousButton = HIGH;
int stableButton = LOW; // Require a stable release after boot before arming.
uint32_t buttonChangedAt = 0;

void ARDUINO_ISR_ATTR onCrossing() {
  // Keep the timestamp path consistent with the test sketch.
  if (!crossed) {
    crossingCycles = ESP.getCycleCount();
    crossed = true;
  }
}

bool measureCapacitance(double &capacitancePf) {
  measuredMedianUs = NAN;
  measuredReferenceMv = 0;
  measuredCpuMhz = 0;
  timingStatus = "clock_error";
  if (!clockReady) return false;
  const uint32_t referenceMv = analogReadMilliVolts(REFERENCE_PIN);
  measuredReferenceMv = referenceMv;
  timingStatus = "reference_error";
  if (referenceMv < 700 || referenceMv > 1100) return false;
  const unsigned cpuMhz = getCpuFrequencyMhz();
  measuredCpuMhz = cpuMhz;
  timingStatus = "clock_error";
  if (cpuMhz != 240) return false;
  timingStatus = "acquisition_error";
  bool complete = true;
  for (unsigned i = 0; i < SAMPLE_COUNT; ++i) {
    gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
    delayMicroseconds(1000);
    elapsedCycles[i] = 0;
    if (digitalRead(COMPARATOR_PIN) != HIGH) {
      complete = false;
      continue;
    }
    crossed = false;
    const uint32_t started = ESP.getCycleCount();
    gpio_set_level((gpio_num_t)CHARGE_PIN, 1);
    while (!crossed && uint32_t(ESP.getCycleCount() - started) < cpuMhz * 1000UL) {}
    if (crossed) elapsedCycles[i] = crossingCycles - started;
    else complete = false;
    gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
  }
  if (!complete) return false; // Never silently drop failed readings.
  for (unsigned i = 1; i < SAMPLE_COUNT; ++i) {
    const uint32_t value = elapsedCycles[i];
    unsigned j = i;
    while (j > 0 && elapsedCycles[j - 1] > value) {
      elapsedCycles[j] = elapsedCycles[j - 1];
      --j;
    }
    elapsedCycles[j] = value;
  }
  const double medianUs = (double(elapsedCycles[31]) + double(elapsedCycles[32])) / (2.0 * cpuMhz);
  measuredMedianUs = medianUs;
  timingStatus = "ok"; // Timing can be valid even when the old pF formula is not.
  capacitancePf = Method1Calibration::capacitancePf(medianUs);
  return isfinite(capacitancePf) && capacitancePf > 0.0;
}

void takeAndReportMeasurement() {
  double capacitancePf = 0.0;
  const bool valid = measureCapacitance(capacitancePf);
  gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
  // Existing numeric failure convention; 0.00 is not a successful measurement.
  if (!valid) capacitancePf = 0.0;
  digitalWrite(STATUS_LED_PIN, LED_ON);
  const uint32_t litAt = millis();
  Serial.println(TEAM_ID);
  Serial.print("Capacitance = ");
  Serial.print(capacitancePf, 2);
  Serial.println(" pF");
  Serial.println();
  Serial.flush();
  while (uint32_t(millis() - litAt) < LED_MS) delay(1);
  digitalWrite(STATUS_LED_PIN, LED_OFF);
  if (calibrationLogging) {
    Serial.print("CAL_DATA,");
    Serial.print(++calibrationRun);
    Serial.print(',');
    if (isfinite(measuredMedianUs)) Serial.print(measuredMedianUs, 6);
    Serial.print(',');
    Serial.print(measuredReferenceMv);
    Serial.print(',');
    Serial.print(measuredCpuMhz);
    Serial.print(',');
    Serial.println(timingStatus);
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LED_OFF);
  pinMode(CHARGE_PIN, OUTPUT);
  digitalWrite(CHARGE_PIN, LOW);
  pinMode(COMPARATOR_PIN, INPUT); // External 2 kOhm pull-up to 3.3 V.
  pinMode(BUTTON_PIN, INPUT_PULLUP); // Button and 100 nF in parallel to GND.
  analogReadResolution(12);
  analogSetPinAttenuation(REFERENCE_PIN, ADC_6db);
  clockReady = setCpuFrequencyMhz(240) && getCpuFrequencyMhz() == 240;
  attachInterrupt(digitalPinToInterrupt(COMPARATOR_PIN), onCrossing, FALLING);
  previousButton = digitalRead(BUTTON_PIN);
  buttonChangedAt = millis();
}

void loop() {
  const int current = digitalRead(BUTTON_PIN);
  if (current != previousButton) {
    previousButton = current;
    buttonChangedAt = millis();
  }
  if (uint32_t(millis() - buttonChangedAt) >= DEBOUNCE_MS && stableButton != current) {
    stableButton = current;
    if (stableButton == HIGH) armed = true;
    else if (armed) {
      armed = false;
      takeAndReportMeasurement();
    }
  }
  // Serial changes logging only; measurements always require the physical button.
  if (Serial.available()) {
    const char command = Serial.read();
    if (command == 'c' && !calibrationLogging) {
      calibrationLogging = true;
      calibrationRun = 0;
      Serial.println("CAL_MODE,on");
      Serial.println("CAL_HEADER,run,median_us,reference_mv,cpu_mhz,status");
    } else if (command == 'q') {
      calibrationLogging = false; // Silent return to exact competition output.
    }
  }
}
