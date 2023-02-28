import csv

divide_n = 3
max_val = 10000
file_i = 1

divide_i = int(max_val / divide_n)
mod = max_val % divide_n

for i in range(1, max_val + 1):
  if (i % divide_i == 0 or i == 1) and i + mod < max_val:
    w = open('train{}.csv'.format(file_i), 'w')
    writer = csv.writer(w)

    file_i += 1

  writer.writerow([i, i])


