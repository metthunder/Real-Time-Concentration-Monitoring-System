void setup() {
  Serial.begin(9600); // Initialize serial communication at 9600 baud
  pinMode(A1, INPUT); // Define A0 as an input
}

void loop() {
  int sensorValue = analogRead(A1); // Read the value from the sensor
  Serial.println(sensorValue);     // Send the value to the laptop via Serial Monitor
  delay(1000); // Wait for 1 second
}