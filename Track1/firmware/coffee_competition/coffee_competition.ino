#include "coffee_sensor.h"
#include "moisture_calibration.h"

constexpr char TEAM_ID[] = "TGR-55";
// 0.00 is a provisional numeric FAILURE sentinel, not a measurement.
// Confirm failure reporting with organizers. Do not submit before calibration.
constexpr double FAILURE_VALUE = 0.0;

void setup() {
  Serial.begin(9600);
  CoffeeSensor::begin();
}

void loop() {
  if (!CoffeeSensor::buttonPressed()) return;
  const CoffeeSensor::Reading r = CoffeeSensor::measure();
  double moisture = FAILURE_VALUE;
  double predicted;
  if (r.valid == CoffeeSensor::COUNT && MoistureCalibration::predict(r.medianUs, predicted))
    moisture = predicted;
  digitalWrite(CoffeeSensor::LED, HIGH);
  const uint32_t start = millis();
  Serial.println(TEAM_ID);
  Serial.print("Moisture Level = ");
  Serial.print(moisture, 2);
  Serial.println(" %");
  Serial.println();
  Serial.flush();
  while (uint32_t(millis()-start) < 200) delay(1);
  digitalWrite(CoffeeSensor::LED, LOW);
}
