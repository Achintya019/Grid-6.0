
#include <Arduino.h>

// Define the IR receiver pin
const int irReceiverPin = 14; // GPIO pin for the IR receiver
int objectCount = 0;          // Counter to track objects
int irState = LOW;            // Current state of the IR receiver (LOW means no object)
int val = 0;                  // Variable to store the IR receiver reading

void setup() {
  // Initialize the IR receiver pin as input
  pinMode(irReceiverPin, INPUT);

  // Start serial communication for debugging
  Serial.begin(115200);

  // Initial message
  Serial.println("Object Counter with IR Sensor Initialized.");
}

void loop() {
  // Read the value from the IR receiver
  val = digitalRead(irReceiverPin);

  // If IR beam is interrupted (object passing through)
  if (val == LOW) {
    if (irState == LOW) {
      // Object detected, increment the counter
      objectCount++;

      // Output the object count
      Serial.print("Object detected! Total count: ");
      Serial.println(objectCount);

      // Update the IR state
      irState = HIGH;
    }
  }
  // If IR beam is restored (no object in front of the sensor)
  else {
    if (irState == HIGH) {
      // Update the IR state back to LOW
      irState = LOW;
    }
  }

  // Small delay to avoid sensor noise or bouncing effects
  delay(100);
}
