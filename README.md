# การฝึกโมเดล
### Step 1: เตรียมข้อมูล
เราจะเริ่มด้วยการแปลงข้อความให้เป็นรูปแบบที่สามารถใช้ในการฝึกโมเดลได้ เช่น การแปลงข้อความเป็นเวคเตอร์โดยใช้ TF-IDF

### Step 2: แบ่งข้อมูลสำหรับฝึกและทดสอบ
เราจะแบ่งข้อมูลออกเป็นส่วนของการฝึก (training) และการทดสอบ (testing)

### Step 3: สร้างและฝึกโมเดล LPN, DNN, SVM
เราจะใช้ Label Propagation Network เพื่อฝึกโมเดล

### Step 4: ทำนายและประเมินผล
เราจะทำนายผลจากชุดข้อมูลทดสอบและประเมินผลลัพธ์

### Step 5: สร้างรายงานและ plot กราฟ
เราจะสร้างรายงานและ plot กราฟเพื่อแสดงผลการทำงานของโมเดล

### Step 6: ปรับปรุงโมเดล
เราจะปรับปรุงโมเดลให้ดีขึ้นโดยใช้วิธีการต่าง ๆ เช่น การปรับพารามิเตอร์, การเปลี่ยนแปลงโมเดล, การเพิ่มข้อมูลใหม่ เป็นต้น

### Step 7: Deploy โมเดล (Web Application)
```python
python src/app.py
```