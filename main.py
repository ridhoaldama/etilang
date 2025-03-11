import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv

# Source
SOURCE_VIDEO = 'tes.mp4'
SOURCE_MODEL = 'best.pt'

# Konfigurasi variabel (threshold dan parameter gambar)
MOTOR_CONF_THRESHOLD = 0.4
WITHOUT_HELMET_CONF_THRESHOLD = 0.3
TNKB_CONF_THRESHOLD_LOW = 0.2
TNKB_CONF_THRESHOLD_HIGH = 0.5
BOUNDING_BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) != 0 else 0

def is_inside(inner_box, outer_box):
    x1, y1, x2, y2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (ox1 <= center_x <= ox2) and (oy1 <= center_y <= oy2)

# Fungsi untuk memperbarui file CSV secara real time dengan mempertahankan data lama
def update_csv(new_data, old_data, csv_file='pelanggar.csv'):
    with open(csv_file, 'w', newline='') as file:
         writer = csv.writer(file)
         writer.writerow(["no", "jenis kendaraan", "img pelanggar", "jenis pelanggaran", "No TNKB"])
         # Tulis data lama yang sudah ada, kemudian data baru dari sesi berjalan
         writer.writerows(old_data + new_data)

# Inisialisasi model
model = YOLO(SOURCE_MODEL)

# Mapping class dan warna
class_names = model.names
motor_class = None
without_helmet_class = None
tnkb_class = None

for idx, name in class_names.items():
    if 'motor' in name.lower():
        motor_class = idx
    elif 'without' in name.lower():
        without_helmet_class = idx
    elif 'tnkb' in name.lower():
        tnkb_class = idx

display_names = {
    motor_class: "Motor",
    without_helmet_class: "Tidak Memakai Helm",
    tnkb_class: "TNKB"
}

colors = {
    "Motor": (255, 0, 0),
    "Tidak Memakai Helm": (0, 0, 255),
    "TNKB": (0, 255, 0)
}

text_color = (0, 0, 0)
background_color = (200, 200, 200)

# Buat folder untuk menyimpan foto pelanggar jika belum ada
os.makedirs("image_pelanggar", exist_ok=True)

# Inisialisasi list untuk menyimpan informasi pelanggar pada sesi ini
# tiap elemen: {'box':, 'row_no':, 'motor_file':, 'tnkb_conf':, 'tnkb_file':}
captured_motors = []
csv_data = []

# Baca data lama dari CSV (jika ada) dan tentukan nomor awal
csv_file = 'pelanggar.csv'
old_data = []
start_counter = 0
if os.path.exists(csv_file):
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        old_data = list(reader)
        if old_data:
            try:
                start_counter = max(int(row[0]) for row in old_data if row[0].isdigit())
            except:
                start_counter = 0

# new_counter akan digunakan untuk melanjutkan penomoran
new_counter = start_counter

# Inisialisasi video
cap = cv2.VideoCapture(SOURCE_VIDEO)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    csv_updated = False  # Flag untuk menandakan ada perubahan pada csv_data

    # Deteksi objek
    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    # Kategorikan deteksi
    motors = []
    without_helmets = []
    tnkbs = []

    for box, score, class_id in zip(boxes, scores, class_ids):
        if class_id == motor_class:
            motors.append((box, score))
        elif class_id == without_helmet_class:
            without_helmets.append((box, score))
        elif class_id == tnkb_class:
            tnkbs.append((box, score))

    # Filter motor valid
    valid_motors = []
    for motor in motors:
        m_box, m_conf = motor
        if m_conf < MOTOR_CONF_THRESHOLD:
            continue

        # Cek apakah ada tanpa helm dalam motor
        has_wh = False
        for wh in without_helmets:
            wh_box, wh_conf = wh
            if wh_conf < WITHOUT_HELMET_CONF_THRESHOLD:
                continue
            if is_inside(wh_box, m_box):
                has_wh = True
                break

        if has_wh:
            valid_motors.append(motor)

    # Proses tiap motor valid
    for motor in valid_motors:
        m_box, _ = motor
        x1, y1, x2, y2 = map(int, m_box)
        
        # Gambar kotak motor dan label
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors["Motor"], BOUNDING_BOX_THICKNESS)
        label = display_names[motor_class]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1-th-4), (x1+tw, y1), background_color, -1)
        cv2.putText(frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, text_color, LABEL_FONT_THICKNESS)

        # Cek apakah motor ini sudah pernah dideteksi (berdasarkan IoU)
        motor_found = False
        for captured in captured_motors:
            if compute_iou(m_box, captured['box']) >= 0.5:
                motor_found = True
                # Update TNKB jika ada deteksi baru dengan confidence lebih tinggi
                best_conf = captured['tnkb_conf']
                best_file = captured['tnkb_file']
                for tnkb in tnkbs:
                    t_box, t_conf = tnkb
                    if t_conf < TNKB_CONF_THRESHOLD_LOW:
                        continue
                    if is_inside(t_box, m_box) and t_conf >= TNKB_CONF_THRESHOLD_HIGH and t_conf > best_conf:
                        best_conf = t_conf
                        x1_t, y1_t, x2_t, y2_t = map(int, t_box)
                        tnkb_crop = frame[y1_t:y2_t, x1_t:x2_t]
                        tnkb_resized_crop = cv2.resize(tnkb_crop, (100, 100))
                        tnkb_filename = f'image_pelanggar/tnkb_{captured["row_no"]}.jpg'
                        cv2.imwrite(tnkb_filename, tnkb_resized_crop)
                        best_file = tnkb_filename
                captured['tnkb_conf'] = best_conf
                captured['tnkb_file'] = best_file
                # Update file TNKB pada data CSV (indeks disesuaikan dengan urutan baru pada sesi ini)
                csv_data[captured['row_no'] - (start_counter + 1)][4] = best_file
                csv_updated = True
                break

        # Jika motor belum pernah terdeteksi, buat entry baru
        if not motor_found:
            new_counter += 1
            row_no = new_counter
            crop = frame[y1:y2, x1:x2]
            resized_crop = cv2.resize(crop, (100, 100))
            motor_filename = f'image_pelanggar/motor_{row_no}.jpg'
            cv2.imwrite(motor_filename, resized_crop)
            
            best_conf = 0
            best_file = "unknown"
            for tnkb in tnkbs:
                t_box, t_conf = tnkb
                if t_conf < TNKB_CONF_THRESHOLD_LOW:
                    continue
                if is_inside(t_box, m_box) and t_conf >= TNKB_CONF_THRESHOLD_HIGH and t_conf > best_conf:
                    best_conf = t_conf
                    x1_t, y1_t, x2_t, y2_t = map(int, t_box)
                    tnkb_crop = frame[y1_t:y2_t, x1_t:x2_t]
                    tnkb_resized_crop = cv2.resize(tnkb_crop, (100, 100))
                    tnkb_filename = f'image_pelanggar/tnkb_{row_no}.jpg'
                    cv2.imwrite(tnkb_filename, tnkb_resized_crop)
                    best_file = tnkb_filename

            new_motor = {
                'box': m_box,
                'row_no': row_no,
                'motor_file': motor_filename,
                'tnkb_conf': best_conf,
                'tnkb_file': best_file
            }
            captured_motors.append(new_motor)
            csv_data.append([row_no, display_names[motor_class], motor_filename, display_names[without_helmet_class], best_file])
            csv_updated = True
        
        # Gambar bounding box untuk tanpa helm yang ada di dalam motor
        for wh in without_helmets:
            wh_box, wh_conf = wh
            if wh_conf < WITHOUT_HELMET_CONF_THRESHOLD:
                continue
            if is_inside(wh_box, m_box):
                x1wh, y1wh, x2wh, y2wh = map(int, wh_box)
                cv2.rectangle(frame, (x1wh, y1wh), (x2wh, y2wh), colors["Tidak Memakai Helm"], BOUNDING_BOX_THICKNESS)
                label_wh = display_names[without_helmet_class]
                (tw, th), _ = cv2.getTextSize(label_wh, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
                cv2.rectangle(frame, (x1wh, y1wh-th-4), (x1wh+tw, y1wh), background_color, -1)
                cv2.putText(frame, label_wh, (x1wh, y1wh-4), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, text_color, LABEL_FONT_THICKNESS)

    # Gambar bounding box TNKB secara terpisah (untuk tampilan)
    # Hanya gambar TNKB jika berada di dalam motor valid (yang memiliki label Without helmet)
    for tnkb in tnkbs:
        t_box, t_conf = tnkb
        if t_conf < TNKB_CONF_THRESHOLD_LOW:
            continue
        inside_valid_motor = False
        for motor in valid_motors:
            m_box, _ = motor
            if is_inside(t_box, m_box):
                inside_valid_motor = True
                break
        if not inside_valid_motor:
            continue
        x1_t, y1_t, x2_t, y2_t = map(int, t_box)
        cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), colors["TNKB"], BOUNDING_BOX_THICKNESS)
        label_tnkb = display_names[tnkb_class]
        (tw, th), _ = cv2.getTextSize(label_tnkb, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
        cv2.rectangle(frame, (x1_t, y1_t-th-4), (x1_t+tw, y1_t), background_color, -1)
        cv2.putText(frame, label_tnkb, (x1_t, y1_t-4), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, text_color, LABEL_FONT_THICKNESS)

    # Jika ada perubahan data, perbarui file CSV secara real time dengan menambahkan data baru ke data lama
    if csv_updated:
        update_csv(csv_data, old_data, csv_file)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
