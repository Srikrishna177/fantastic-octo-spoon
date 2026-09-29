import sys
sys.stdout.encoding
print(f"Console encoding: {sys.stdout.encoding}")
print(f"File system encoding: {sys.getfilesystemencoding()}")

test_str = "Master Full-Stack Docker & CI/CD – Build a Production-Ready Pipeline"
test_str_clean = test_str.replace('–', '-')

print(f"Original: {repr(test_str)}")
print(f"Cleaned: {repr(test_str_clean)}")

with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write(test_str_clean)

print("Written successfully!")
