import csv

# Open the CSV file and read it into a list of lists
with open('CLIP_VITB32.csv', 'r') as f:
  reader = csv.reader(f, delimiter='\n')
  data = list(reader)

# Print the data
# for d in data:
#     print(len(d))

print(data[0])