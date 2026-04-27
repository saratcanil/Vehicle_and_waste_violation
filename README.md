# 🚮 Night-Time Waste Dumping Detection using PTZ Camera

## 📌 Overview
This project focuses on detecting **illegal waste dumping at night** using a combination of:
- PTZ CCTV cameras
- Low-light imaging (no IR)
- AI-based detection (YOLO + motion logic)

## The system detects:
-  Vehicles stopping in restricted zones
-  Waste dumping action
-  Captures number plate (when possible)
-  triple riding
-  helmet detection


## 🎯 Objectives
- Detect dumping activity in low-light conditions
- Avoid IR-based imaging (to preserve scene details)
- Capture both **action + evidence**
- Minimize false positives

---

## 🧠 System Architecture
<img width="1320" height="1600" alt="WhatsApp Image 2026-01-07 at 16 16 07" src="https://github.com/user-attachments/assets/6023fecb-054f-4e68-aeba-361f231939b7" />

---

## 📷 Hardware Requirements

- PTZ Camera (with optical zoom)
- Low-light sensor (Sony STARVIS recommended)
- White light illumination (no IR)
- Mounting height: 3–6 meters

---

## ⚙️ Software Stack

- Python
- OpenCV
- YOLO (v11)
- NumPy
- OCR for number plates

---



## 🌙 Night Optimization

- Use **full-color low-light cameras**
- Avoid IR (causes reflection issues)
- Use **soft white lighting**
- Manual camera settings:
  - Shutter: 1/100 – 1/250
  - WDR: OFF

---

## 📊 Limitations

- Not suitable for high-speed vehicles
- Requires minimum ambient lighting
- PTZ alone is not sufficient for detection

---

## 📸 Sample Outputs
<img width="768" height="520" alt="WhatsApp Image 2026-01-19 at 16 26 19" src="https://github.com/user-attachments/assets/d4562b4b-461e-455a-ad62-399171911c32" />
<img width="1600" height="892" alt="WhatsApp Image 2026-04-27 at 21 08 40" src="https://github.com/user-attachments/assets/a2bd6f89-fcb0-41b1-b4bd-c87f4ef9c579" />

---

## 🚀 Future Improvements

- Better action recognition models
- Real-time alert system (SMS/Email)
- Integration with municipal dashboards

