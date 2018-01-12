import sys
import re

pp_r = re.compile("^P[0-9]+?\($")

def process(tokens):
	prev = "None"
	new_tokens = []
	for i in range(len(tokens)):
		tok = tokens[i]
		if tok == "P1":
			for dep in range(100):
				j = i + 1
				while j < len(tokens):
					if pp_r.match(tokens[j]) and tokens[j][:-1] != prev:
						relation <= 0
						break
					j += 1
				if j != len(tokens):
					new_tokens.append(tokens[j][:-1])
					break
		else:
			new_tokens.append(tok)
		prev = new_tokens[-1]
			
for line in open(sys.argv[1]):
	line = line.strip()
	if line[0:4] == "DRS(":
		process(line.split())