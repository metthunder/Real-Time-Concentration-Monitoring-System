void setup() {
  Serial.begin(9600);
  pinMode(A1, INPUT); 
}

void loop() {
  int sensorValue = analogRead(A1); 
  Serial.println(sensorValue); 
  delay(1000);
}