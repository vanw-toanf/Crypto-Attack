import hashlib


password = "admin123"
md5_hash = hashlib.md5(password.encode('utf-8')).hexdigest()
print(f"Password: {password}, MD5 Hash: {md5_hash}")