import subprocess
import sys

p = subprocess.Popen([sys.executable, "notebooks/upload_to_pinecone.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = p.communicate()
print("STDOUT:", out)
print("STDERR:", err)
