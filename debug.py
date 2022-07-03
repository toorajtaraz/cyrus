# Open a file that has this format
# x <int> y <int> [before|after]
# find "before" incidents that do not have "after" incident


before = []
after = []
file_path = ".\\out.txt"
with open(file_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("x"):
            x = int(line.split()[1])
            y = int(line.split()[3])
            if line.split()[4] == "before":
                before.append((x, y))
            else:
                after.append((x, y))

# Find tuples that are in before but not in after
print(len(list(set(before))), len(list(set(after))))
for t in before:
    if t not in after:
        print(t)