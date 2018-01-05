import sys

for line in open(sys.argv[1]):
	line = line.strip().split()
	pnum = 1
	knum = 1
	for i in range(len(line)):
		if line[i] == 'P1(':
			line[i] = 'P'+str(pnum)+'('
			pnum += 1
		elif line[i] == 'K1(':
			line[i] = 'K'+str(knum)+'('
			knum += 1
	print " ".join(line)
