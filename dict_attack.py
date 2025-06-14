import hashlib
import csv

def dictionary_attack(target_hash, dictionary_file):
    """
    Thực hiện tấn công từ điển vào một mã hash MD5.
    Returns:
        str: Mật khẩu tìm thấy, hoặc None nếu không tìm thấy.
    """
    print(f"[*] Bắt đầu tấn công mã hash: {target_hash}")
    print(f"[*] Sử dụng từ điển: {dictionary_file}")

    try:
        with open(dictionary_file, 'r', encoding='utf-8', errors='ignore') as file:
            next(file)
            
            reader = csv.reader(file)
            for row in reader:
                if not row:  
                    continue
                password = row[0]

                calculated_hash = hashlib.md5(password.encode()).hexdigest()
                
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

if __name__ == "__main__":
    hash_to_crack = input("Nhập mã hash MD5 cần tìm mật khẩu: ")
    
    dictionary_path = "dataset/rockyou.csv" 
    
    found_password = dictionary_attack(hash_to_crack, dictionary_path)
    
    if found_password:
        print(f"    - Mật khẩu là: {found_password}")
        print(f"    - Mã MD5: {hash_to_crack}")