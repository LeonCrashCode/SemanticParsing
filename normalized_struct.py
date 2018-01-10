import sys
import re

pp_r = re.compile("P[0-9]+?\(")

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

	newline = []
	flag = True
	stack = []
	i = 0
	while i < len(line):
		tok = line[i]
		if pp_r.match(tok) and int(tok[1:-1]) >= 11:
			relation = 1
			while True:
				if relation == 0:
					break
				if line[i] == ")":
					relation -= 1
				else:
					relation += 1
				i += 1
		else:
			newline.append(tok)
			i += 1
	print " ".join(newline[:-1])
