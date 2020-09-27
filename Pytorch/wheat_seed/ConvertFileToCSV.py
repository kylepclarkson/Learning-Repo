import csv


file = open('wheat.txt', 'r')
content_to_write = []

for line in file:
    content_to_write.append(line.split('\t'))

with open('wheat.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    [writer.writerow([str(item).strip() for item in entry if len(item)>0]) for entry in content_to_write]

