with open('./A196_task2.txt', 'r') as f:
    data = f.readlines()

print(len(data))
count = 0
for data_line in data:
    if data_line[:11].lower() == data_line[-12:].strip():
        count += 1
print(count)