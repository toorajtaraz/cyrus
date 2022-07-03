# Open a file that has this format
# <int> <int>

file_path = ".\\out.txt"
all_the_tuples = []
with open(file_path, "r") as f:
    lines = f.readlines()
    # See if there are any tuple in the file that are repeated more than once
    for line in lines:
        line = line.strip()
        line = line.split(" ")
        line = tuple(line)
        if line in all_the_tuples:
            print("Repeated tuple: ", line)
            continue
        all_the_tuples.append(line)