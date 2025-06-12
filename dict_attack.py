import hashlib
import csv

def dictionary_attack(target_hash, dictionary_file):
    """
    Thực hiện tấn công từ điển vào một mã hash MD5.

    Args:
        target_hash (str): Mã MD5 cần tìm mật khẩu.
        dictionary_file (str): Đường dẫn đến file CSV chứa mật khẩu.
                                Định dạng: password,md5_hash

    Returns:
        str: Mật khẩu tìm thấy, hoặc None nếu không tìm thấy.
    """
    print(f"[*] Bắt đầu tấn công mã hash: {target_hash}")
    print(f"[*] Sử dụng từ điển: {dictionary_file}")

    try:
        with open(dictionary_file, 'r', encoding='utf-8', errors='ignore') as file:
            # Bỏ qua dòng tiêu đề 'password,md5_hash'
            next(file)
            
            # Đọc từng dòng trong file csv
            reader = csv.reader(file)
            for row in reader:
                if not row:  # Bỏ qua các dòng trống
                    continue
                password = row[0]
                
                # Tính toán mã MD5 cho mật khẩu
                # encode() để chuyển chuỗi thành bytes, yêu cầu của hashlib
                calculated_hash = hashlib.md5(password.encode()).hexdigest()
                
                # So sánh với mã hash mục tiêu
                if calculated_hash == target_hash:
                    print(f"\n[+] THÀNH CÔNG! Mật khẩu được tìm thấy.")
                    return password
                    
    except FileNotFoundError:
        print(f"[!] LỖI: Không tìm thấy file {dictionary_file}")
        return None
    except Exception as e:
        print(f"[!] Đã xảy ra lỗi: {e}")
        return None
        
    print(f"\n[-] THẤT BẠI. Không tìm thấy mật khẩu trong từ điển.")
    return None

# --- Phần thực thi ---
if __name__ == "__main__":
    # Mã hash MD5 của "12345" để thử nghiệm
    # hash_to_crack = "827ccb0eea8a706c4c34a16891f84e7b"
    hash_to_crack = input("Nhập mã hash MD5 cần tìm mật khẩu: ")
    
    # Tên file từ điển của bạn
    # Hãy đảm bảo file này nằm cùng thư mục với file Python
    # hoặc cung cấp đường dẫn đầy đủ.
    dictionary_path = "dataset/rockyou.csv" 
    
    found_password = dictionary_attack(hash_to_crack, dictionary_path)
    
    if found_password:
        print(f"    - Mật khẩu là: {found_password}")
        print(f"    - Mã MD5: {hash_to_crack}")