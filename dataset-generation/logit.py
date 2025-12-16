
def log(res):
    with open("loggs.txt", "a") as f:
        f.write(str(res))
        f.write("\n")
    return