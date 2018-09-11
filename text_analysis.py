path = '/Users/jack/Downloads/train.txt'

fs = open(path, 'r+')
s = set()
for line in fs.readlines():
    s.update(line.split(',')[1:])
print(len(s))
fs.close()
