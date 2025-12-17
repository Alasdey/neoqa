
from datetime import datetime

def log(res):
    with open("loggs.txt", "a") as f:
        f.write(str(res))
        f.write("\n")
    return

def logg_llm(prompt: str, response: str):
    with open("temp6.txt", "a") as f:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(timestamp_str + "\n\n")
        f.write(prompt)
        f.write("\n\n---------------------------------------------------------------------------\n\n")
        f.write(response)
        f.write("\n\n===========================================================================\n\n")
        f.close()
