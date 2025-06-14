import csv
import random

SOURCE_FILE = 'dataset/rockyou.csv'
TRAIN_FILE = 'dataset/train.csv'
TEST_FILE = 'dataset/test.csv'
TEST_SPLIT = 0.1

print(f"[*] Đang đọc file gốc: {SOURCE_FILE}")
with open(SOURCE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    header = next(reader)
    lines = [row for row in reader if row]

print(f"[*] Trộn và chia dữ liệu...")
random.shuffle(lines)
split_index = int(len(lines) * (1 - TEST_SPLIT))
train_lines = lines[:split_index]
test_lines = lines[split_index:]

print(f"[*] Ghi {len(train_lines)} dòng vào file huấn luyện: {TRAIN_FILE}")
with open(TRAIN_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(train_lines)

print(f"[*] Ghi {len(test_lines)} dòng vào file kiểm tra: {TEST_FILE}")
with open(TEST_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(test_lines)

print("[+] Hoàn tất!")