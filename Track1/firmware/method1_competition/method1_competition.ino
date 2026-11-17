#include <Arduino.h>
#include <math.h>
#include "driver/gpio.h"

// Capacitance contest only. DOIT ESP32 DEVKIT V1 / original ESP32.
constexpr char TEAM_ID[] = "TGR-55";
constexpr int CHARGE_PIN = 25;
constexpr int COMPARATOR_PIN = 27;
constexpr int REFERENCE_PIN = 34;
constexpr int BUTTON_PIN = 26;
constexpr int STATUS_LED_PIN = 2;
constexpr int LED_ON = HIGH; // Confirm polarity on the physical board.
constexpr int LED_OFF = LOW;
constexpr unsigned SAMPLE_COUNT = 64;
constexpr uint32_t DEBOUNCE_MS = 30;
constexpr uint32_t LED_MS = 200;

// Frozen round-1 fit, applied to the median of a complete 64-reading batch.
// Validated nominal range: 5-20 pF; same circuit, ~0.9 V reference, 240 MHz.
constexpr double CAL_SLOPE = 3.0739224418808453;
constexpr double CAL_INTERCEPT = -18.352037372425478;

volatile bool crossed = false;
volatile uint32_t crossingCycles = 0;
uint32_t elapsedCycles[SAMPLE_COUNT];
bool clockReady = false;
bool armed = false;
int previousButton = HIGH;
int stableButton = LOW; // Require a debounced release after boot.
uint32_t buttonChangedAt = 0;

void ARDUINO_ISR_ATTR onCrossing() {
  // Preserve the timing path used in the bench calibration.
  if (!crossed) {
    crossingCycles = ESP.getCycleCount();
    crossed = true;
  }
}

bool measureCapacitance(double &capacitancePf) {
  if (!clockReady) return false;
  // Same single reference read as the bench sketch; no reads during timing.
  const uint32_t referenceMv = analogReadMilliVolts(REFERENCE_PIN);
  if (referenceMv < 700 || referenceMv > 1100) return false;
  const unsigned cpuMhz = getCpuFrequencyMhz();
  if (cpuMhz != 240) return false;
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
  // Do not change the estimator by silently dropping failed readings.
  if (!complete) return false;
  for (unsigned i = 1; i < SAMPLE_COUNT; ++i) {
    uint32_t value = elapsedCycles[i];
    unsigned j = i;
    while (j > 0 && elapsedCycles[j - 1] > value) {
      elapsedCycles[j] = elapsedCycles[j - 1];
      --j;
    }
    elapsedCycles[j] = value;
  }
  const double medianUs = (double(elapsedCycles[31]) + double(elapsedCycles[32])) / (2.0 * cpuMhz);
  capacitancePf = CAL_SLOPE * medianUs + CAL_INTERCEPT;
  // Do not clamp to reference values or silently reuse the previous result.
  return isfinite(capacitancePf) && capacitancePf > 0.0;
}

void takeAndReportMeasurement() {
  double capacitancePf = 0.0;
  const bool valid = measureCapacitance(capacitancePf);
  gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
  // Failed acquisition reports 0.00 pF, never a fabricated valid measurement.
  // Confirm this numeric failure convention with organizers before handover.
  if (!valid) capacitancePf = 0.0;
  const uint32_t litAt = millis();
  digitalWrite(STATUS_LED_PIN, LED_ON);
  Serial.println(TEAM_ID);
  Serial.print("Capacitance = ");
  Serial.print(capacitancePf, 2);
  Serial.println(" pF");
  Serial.println();
  Serial.flush();
  while (uint32_t(millis() - litAt) < LED_MS) delay(1);
  digitalWrite(STATUS_LED_PIN, LED_OFF);
}

void setup() {
  Serial.begin(9600);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LED_OFF);
  pinMode(CHARGE_PIN, OUTPUT);
  digitalWrite(CHARGE_PIN, LOW);
  pinMode(COMPARATOR_PIN, INPUT); // External 2 kOhm pullup to 3.3 V.
  pinMode(BUTTON_PIN, INPUT_PULLUP); // Button and 0.1 uF in parallel to GND.
  analogReadResolution(12);
  analogSetPinAttenuation(REFERENCE_PIN, ADC_6db);
  clockReady = setCpuFrequencyMhz(240) && getCpuFrequencyMhz() == 240;
  attachInterrupt(digitalPinToInterrupt(COMPARATOR_PIN), onCrossing, FALLING);
  previousButton = digitalRead(BUTTON_PIN);
  buttonChangedAt = millis();
  // No application startup/debug messages on the competition serial channel.
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
  // Serial input commands are intentionally not used to trigger competition reads.
}
