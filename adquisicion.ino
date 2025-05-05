// Adquisición de datos de campo magnético y temperatura

#include <Wire.h>
#include <SPI.h>

// Pines y configuración para RM3100
#define RM3100Address 0x20 // Dirección esclava hexadecimal
#define PIN_DRDY1 8 // Pin Data Ready sensor 1
#define PIN_DRDY2 9 // Pin Data Ready sensor 2

// Valores de registro interno de los RM3100
#define RM3100_REVID_REG 0x36
#define RM3100_POLL_REG 0x00
#define RM3100_CMM_REG 0x01
#define RM3100_STATUS_REG 0x34
#define RM3100_CCX1_REG 0x04
#define RM3100_CCX0_REG 0x05
#define RM3100_RCMX_REG 0x04
#define RM3100_RCMY_REG 0x06
#define RM3100_RCMZ_REG 0x08
#define RM3100_RTMRC_REG 0x8B
#define RM3100_TMRC_REG 0x0B

// Opciones para RM3100
#define initialCC 800 // Cuenta de ciclos
#define singleMode 0 // 0 = modo continuo; 1 = modo de medición única
#define useDRDYPin 1 // 0 = no usar pin DRDY; 1 = usar pin DRDY

//uint8_t revid;
uint16_t cycleCount;
uint16_t cycleCount0;
uint16_t cycleCountX;
uint16_t cycleCountY;
uint16_t cycleCountZ;
uint8_t frecuencia;
float ganancia;

// Pines y configuración para MAX31855
const int csPin = 10;

void setup() {
  // Configuración para MAX31855
  pinMode(csPin, OUTPUT);
  digitalWrite(csPin, HIGH);
  SPI.begin();

  // Configuración para RM3100
  pinMode(PIN_DRDY1, INPUT);
  pinMode(PIN_DRDY2, INPUT);
  Wire.begin();
  Serial.begin(9600);
  delay(100);

  changeCycleCount(initialCC);
  cycleCount = readReg(RM3100_CCX1_REG);
  cycleCount0 = readReg(RM3100_CCX0_REG);
  cycleCountX = readReg(RM3100_RCMX_REG);
  cycleCountY = readReg(RM3100_RCMY_REG);
  cycleCountZ = readReg(RM3100_RCMZ_REG);
  cycleCount = (cycleCount << 8) | readReg(RM3100_CCX0_REG);
  cycleCountX = (cycleCountX << 8) | readReg(RM3100_CCX0_REG);
  cycleCountY = (cycleCountY << 8) | readReg(RM3100_CCX0_REG);
  cycleCountZ = (cycleCountZ << 8) | readReg(RM3100_CCX0_REG);

  // Ganancia
  ganancia = (0.3671 * (float)cycleCount) + 1.5;

  // Modo de medición
  if (singleMode){
    writeReg(RM3100_CMM_REG, 0);
    writeReg(RM3100_POLL_REG, 0x70);
  }
  else {
    writeReg(RM3100_CMM_REG, 0x79);
  }

  // Frecuencia de sampleo
  writeReg(RM3100_TMRC_REG, 0x92);
  delay(10);
  frecuencia = readReg(RM3100_RTMRC_REG);

  // Primera fila con los nombres de las columnas
  Serial.print("Tiempo1(ms) ");
  Serial.print("Tiempo2(ms) ");
  Serial.print("Bx1(cuentas) ");
  Serial.print("Bx1(muT) ");
  Serial.print("By1(cuentas) ");
  Serial.print("By1(muT) ");
  Serial.print("Bz1(cuentas) ");
  Serial.print("Bz1(muT) ");
  Serial.print("Bx2(cuentas) ");
  Serial.print("Bx2(muT) ");
  Serial.print("By2(cuentas) ");
  Serial.print("By2(muT) ");
  Serial.print("Bz2(cuentas) ");
  Serial.print("Bz2(muT) ");
  Serial.println("Temperatura(C)");
}

void loop() {
  // Leer y mostrar datos de campo magnético del primer RM3100
  long x1 = 0, y1 = 0, z1 = 0;
  uint8_t x12, x11, x10, y12, y11, y10, z12, z11, z10;
  unsigned long tiempo_1;
  selectChannel(6); // Canal del primer sensor

  // Esperar a que los datos estén listos
  // Usando pin Data Ready
  if (useDRDYPin) {
    while(digitalRead(PIN_DRDY1) == LOW);
  }
  // Sin usar pin Data Ready
  else {
    while((readReg(RM3100_STATUS_REG) & 0x80) != 0x80);
  }

  Wire.beginTransmission(RM3100Address);
  Wire.write(0x24);
  Wire.endTransmission();

  // Solicitar 9 bytes
  Wire.requestFrom(RM3100Address, 9);
  if(Wire.available() == 9) {
    tiempo_1 = millis();

    x12 = Wire.read();
    x11 = Wire.read();
    x10 = Wire.read();
    
    y12 = Wire.read();
    y11 = Wire.read();
    y10 = Wire.read();
    
    z12 = Wire.read();
    z11 = Wire.read();
    z10 = Wire.read();
  }

  // Manipulación especial de bits
  if (x12 & 0x80){
    x1 = 0xFF;
  }
  if (y12 & 0x80){
    y1 = 0xFF;
  }
  if (z12 & 0x80){
    z1 = 0xFF;
  }

  // Convertir resultados a 32 bits
  x1 = (x1 * 256 * 256 * 256) | (int32_t)(x12) * 256 * 256 | (uint16_t)(x11) * 256 | x10;
  y1 = (y1 * 256 * 256 * 256) | (int32_t)(y12) * 256 * 256 | (uint16_t)(y11) * 256 | y10;
  z1 = (z1 * 256 * 256 * 256) | (int32_t)(z12) * 256 * 256 | (uint16_t)(z11) * 256 | z10;

  // Leer y mostrar datos de campo magnético del segundo RM3100
  long x2 = 0, y2 = 0, z2 = 0;
  uint8_t x22, x21, x20, y22, y21, y20, z22, z21, z20;
  unsigned long tiempo_2;
  selectChannel(7); // Canal del segundo sensor

  // Esperar a que los datos estén listos
  if (useDRDYPin) {
    while(digitalRead(PIN_DRDY2) == LOW);
  } else {
    while((readReg(RM3100_STATUS_REG) & 0x80) != 0x80);
  }

  Wire.beginTransmission(RM3100Address);
  Wire.write(0x24);
  Wire.endTransmission();

  // Solicitar 9 bytes
  Wire.requestFrom(RM3100Address, 9);
  if(Wire.available() == 9) {
    tiempo_2 = millis();

    x22 = Wire.read();
    x21 = Wire.read();
    x20 = Wire.read();
    
    y22 = Wire.read();
    y21 = Wire.read();
    y20 = Wire.read();
    
    z22 = Wire.read();
    z21 = Wire.read();
    z20 = Wire.read();
  }

  if (x22 & 0x80){
    x2 = 0xFF;
  }
  if (y22 & 0x80){
    y2 = 0xFF;
  }
  if (z22 & 0x80){
    z2 = 0xFF;
  }

  x2 = (x2 * 256 * 256 * 256) | (int32_t)(x22) * 256 * 256 | (uint16_t)(x21) * 256 | x20;
  y2 = (y2 * 256 * 256 * 256) | (int32_t)(y22) * 256 * 256 | (uint16_t)(y21) * 256 | y20;
  z2 = (z2 * 256 * 256 * 256) | (int32_t)(z22) * 256 * 256 | (uint16_t)(z21) * 256 | z20;

  // Conversión de cuentas a muT
  float x1_ut = (float)(x1)/ganancia;
  float y1_ut = (float)(y1)/ganancia;
  float z1_ut = (float)(z1)/ganancia;
  float x2_ut = (float)(x2)/ganancia;
  float y2_ut = (float)(y2)/ganancia;
  float z2_ut = (float)(z2)/ganancia;

  // Envío de datos por serie
  Serial.print(tiempo_1);
  Serial.print(" ");
  Serial.print(tiempo_2);
  Serial.print(" ");
  Serial.print(x1);
  Serial.print(" ");
  Serial.print(x1_ut, 3); // 3 decimales
  Serial.print(" ");
  Serial.print(y1);
  Serial.print(" ");
  Serial.print(y1_ut, 3);
  Serial.print(" ");
  Serial.print(z1);
  Serial.print(" ");
  Serial.print(z1_ut, 3);
  Serial.print(" ");
  Serial.print(x2);
  Serial.print(" ");
  Serial.print(x2_ut, 3);
  Serial.print(" ");
  Serial.print(y2);
  Serial.print(" ");
  Serial.print(y2_ut, 3);
  Serial.print(" ");
  Serial.print(z2);
  Serial.print(" ");
  Serial.print(z2_ut, 3);
  Serial.print(" ");

  // Leer y mostrar datos de temperatura del MAX31855
  uint32_t data = readMAX31855();

  // Verifica si hay algún error
  if (data & 0x00010000) {
    Serial.print("Error de termocupla");
    Serial.println();
  } else if (data & 0x00000001) {
    Serial.print("Termocupla desconectada");
    Serial.println();
  } else if (data & 0x00000002) {
    Serial.print("Cortocircuito a VCC");
    Serial.println();
  } else if (data & 0x00000004) {
    Serial.print("Cortocircuito a GND");
    Serial.println();
  } else {
    int16_t tempRaw = (data >> 18) & 0x3FFF; // bits D31 a D18
    if (tempRaw & 0x2000) {
      tempRaw |= 0xC000; // Extiende el signo si es necesario
    }
    float tempCelsius = tempRaw * 0.25;  // Conversión a grados Celsius
    Serial.print(tempCelsius, 2); // Valor con 2 decimales
    Serial.println();
  }
}

// Función de selección de canal
void selectChannel(uint8_t channel) {
  Wire.beginTransmission(0x71);  // Dirección del TCA9548A (A0 a 3.3V, y A1 y A2 a GND)
  Wire.write(1 << channel);      // Seleccionar el canal adecuado
  Wire.endTransmission();
}

// Funciones para el RM3100
uint8_t readReg(uint8_t addr){
  uint8_t data = 0;
  
  Wire.beginTransmission(RM3100Address);
  Wire.write(addr);
  Wire.endTransmission();
  delay(100);

  Wire.requestFrom(RM3100Address, 1);
  if(Wire.available() == 1) {
    data = Wire.read();
  }
  return data;
}

void writeReg(uint8_t addr, uint8_t data){
  Wire.beginTransmission(RM3100Address);
  Wire.write(addr);
  Wire.write(data);
  Wire.endTransmission();
}

void changeCycleCount(uint16_t newCC){
  uint8_t CCMSB = (newCC & 0xFF00) >> 8;
  uint8_t CCLSB = newCC & 0xFF;
  
  Wire.beginTransmission(RM3100Address);
  Wire.write(RM3100_CCX1_REG);
  Wire.write(CCMSB);
  Wire.write(CCLSB);
  Wire.write(CCMSB);
  Wire.write(CCLSB);
  Wire.write(CCMSB);
  Wire.write(CCLSB);    
  Wire.endTransmission();  
}

// Función para leer datos del MAX31855
uint32_t readMAX31855() {
  digitalWrite(csPin, LOW);  // Activa el CS
  delayMicroseconds(1);
  
  // Leer 32 bits de datos desde el MAX31855
  uint32_t data = SPI.transfer(0x00);
  data <<= 8;
  data |= SPI.transfer(0x00);
  data <<= 8;
  data |= SPI.transfer(0x00);
  data <<= 8;
  data |= SPI.transfer(0x00);
  
  digitalWrite(csPin, HIGH);  // Desactiva el CS
  
  return data;
}
