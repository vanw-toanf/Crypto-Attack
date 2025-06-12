import hashlib

with open("dataset/rockyou.txt", "r", encoding="latin-1") as infile, open("dataset/rockyou.csv", "w") as outfile:
    outfile.write("password,md5_hash\n")
    for line in infile:
        password = line.strip()
        if password:  
            md5_hash = hashlib.md5(password.encode('utf-8')).hexdigest()
            outfile.write(f"{password},{md5_hash}\n")
