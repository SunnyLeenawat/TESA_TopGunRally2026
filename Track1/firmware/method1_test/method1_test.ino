#include <Arduino.h>
#include <math.h>
#include "driver/gpio.h"

// ESP32-WROOM-32 bench experiment: LM393 inverting threshold timing.
// Raw timing plus the same frozen round-1 calibration as the competition sketch.
constexpr int CHARGE_PIN = 25;
constexpr int COMPARATOR_PIN = 27;
// LM393 OUT1 -> GPIO27, with external 2 kOhm pullup to 3.3 V.
// IN+ = reference, IN- = charging capacitor: capture HIGH -> LOW.
constexpr int REFERENCE_PIN = 34;
constexpr int BUTTON_PIN = 26;
// Momentary switch and 0.1 uF (100 nF) capacitor in parallel to GND.
// Internal pullup and the 30 ms software debounce remain enabled.
constexpr unsigned SAMPLE_COUNT = 64;
constexpr double CAL_SLOPE = 3.0739224418808453;
constexpr double CAL_INTERCEPT = -18.352037372425478;
volatile bool crossed = false;
volatile uint32_t crossingCycles = 0;
uint32_t elapsedCycles[SAMPLE_COUNT];
uint8_t statusCode[SAMPLE_COUNT]; // 0 valid, 1 bad initial state, 2 timeout
unsigned long runId = 0;

void ARDUINO_ISR_ATTR onCrossing() {
  if (!crossed) {
    crossingCycles = ESP.getCycleCount();
    crossed = true;
  }
}

void reportCapacitance(unsigned cpuMhz) {
  unsigned validCount = 0;
  uint32_t sorted[SAMPLE_COUNT];
  for (unsigned i = 0; i < SAMPLE_COUNT; ++i) {
    validCount += statusCode[i] == 0;
    sorted[i] = elapsedCycles[i];
  }
  Serial.printf("SUMMARY run=%lu valid=%u/64 cpu_mhz=%u\n", runId, validCount, cpuMhz);
  if (validCount != SAMPLE_COUNT || cpuMhz != 240) {
    Serial.println("SUMMARY Capacitance = INVALID (requires 64 ok readings at 240 MHz)");
    return;
  }
  // Sort a copy so raw CSV readings remain in acquisition order.
  for (unsigned i = 1; i < SAMPLE_COUNT; ++i) {
    uint32_t value = sorted[i];
    unsigned j = i;
    while (j > 0 && sorted[j - 1] > value) {
      sorted[j] = sorted[j - 1];
      --j;
    }
    sorted[j] = value;
  }
  const double medianUs = (double(sorted[31]) + double(sorted[32])) / (2.0 * cpuMhz);
  const double capacitancePf = CAL_SLOPE * medianUs + CAL_INTERCEPT;
  Serial.print("SUMMARY median_elapsed_us = ");
  Serial.println(medianUs, 4);
  if (!isfinite(capacitancePf) || capacitancePf <= 0.0) {
    Serial.println("SUMMARY Capacitance = INVALID (nonpositive or nonfinite result)");
    return;
  }
  Serial.print("SUMMARY Capacitance = ");
  Serial.print(capacitancePf, 2);
  Serial.println(" pF");
}

void measure() {
  const uint32_t referenceMv = analogReadMilliVolts(REFERENCE_PIN);
  if (referenceMv < 700 || referenceMv > 1100) {
    Serial.println("SET_REFERENCE: adjust pot to about 900 mV; send r to check");
    return;
  }
  const unsigned cpuMhz = getCpuFrequencyMhz();
  for (unsigned i = 0; i < SAMPLE_COUNT; ++i) {
    gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
    delayMicroseconds(1000);
    elapsedCycles[i] = 0;
    statusCode[i] = 0;
    if (digitalRead(COMPARATOR_PIN) != HIGH) {
      statusCode[i] = 1;
      continue;
    }
    crossed = false;
    const uint32_t started = ESP.getCycleCount();
    gpio_set_level((gpio_num_t)CHARGE_PIN, 1);
    while (!crossed && uint32_t(ESP.getCycleCount() - started) < cpuMhz * 1000UL) {}
    if (crossed) elapsedCycles[i] = crossingCycles - started;
    else statusCode[i] = 2;
    gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
  }
  ++runId;
  // Summary precedes the CSV so capture saves it even for the final batch.
  // All calculation and serial output happen after acquisition is complete.
  reportCapacitance(cpuMhz);
  Serial.println("run,reading,cycles,elapsed_us,reference_mv,cpu_mhz,status");
  for (unsigned i = 0; i < SAMPLE_COUNT; ++i) {
    Serial.printf("%lu,%u,%lu,", runId, i + 1, (unsigned long)elapsedCycles[i]);
    if (statusCode[i] == 0) Serial.print(double(elapsedCycles[i]) / cpuMhz, 4);
    Serial.printf(",%lu,%u,%s\n", (unsigned long)referenceMv, cpuMhz,
      statusCode[i] == 0 ? "ok" : statusCode[i] == 1 ? "initial_low" : "timeout");
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(CHARGE_PIN, OUTPUT);
  digitalWrite(CHARGE_PIN, LOW);
  pinMode(COMPARATOR_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  analogReadResolution(12);
  analogSetPinAttenuation(REFERENCE_PIN, ADC_6db);
  attachInterrupt(digitalPinToInterrupt(COMPARATOR_PIN), onCrossing, FALLING);
  Serial.println("M1 bench: r = reference mV, m = 64 readings + calibrated pF; button also starts a batch");
}

void loop() {
  static int previous = HIGH, stable = HIGH;
  static unsigned long changedAt = 0;
  int current = digitalRead(BUTTON_PIN);
  if (current != previous) { previous = current; changedAt = millis(); }
  if (millis() - changedAt >= 30 && stable != current) {
    stable = current;
    if (stable == LOW) measure();
  }
  if (Serial.available()) {
    char command = Serial.read();
    if (command == 'r') Serial.println(analogReadMilliVolts(REFERENCE_PIN));
    if (command == 'm' && stable == HIGH) measure();
  }
}
