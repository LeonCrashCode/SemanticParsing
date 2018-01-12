import sys
import re

kp_r = re.compile("^K[0-9]+?\($")
pp_r = re.compile("^P[0-9]+?\($")

kp = re.compile("^K[0-9]+?$")
pp = re.compile("^P[0-9]+?$")
xp = re.compile("^X[0-9]+?$")
sp = re.compile("^S[0-9]+?$")
ep = re.compile("^E[0-9]+?$")

special = ["NOT(", "POS(", "NEC(", "DUPLEX(", "IMP(", "OR("]

def get_childs(tokens):
	stack = 0
	childs = []
	childs.append(0)
	for i in range(len(tokens)):
		if tokens[i][-1] == "(":
			stack += 1
		elif tokens[i] == ")":
			stack -= 1
		if stack == 0:
			childs.append(i+1)
	return childs
def reorder(childs, tokens):
	if tokens[0] == "DRS(":
		return reorder_drs(childs, tokens)
	elif tokens[0] == "SDRS(":
		return reorder_sdrs(childs, tokens)
	else:
		return reorder_normal(childs, tokens)
def reorder_normal(childs, tokens):
	reorder_childs = []
	for i in range(len(childs)-1):
		reorder_childs.append((childs[i]+1, childs[i+1]))
	return reorder_childs

def reorder_drs(childs, tokens):
	reorder_childs = []
	for i in range(len(childs)-1):
		if pp_r.match(tokens[childs[i]+1]) or (tokens[childs[i]+1] in special):
			pass
		else:
			reorder_childs.append((childs[i]+1, childs[i+1]))

	for i in range(len(childs)-1):
		if pp_r.match(tokens[childs[i]+1]) or (tokens[childs[i]+1] in special):
			reorder_childs.append((childs[i]+1, childs[i+1]))
		else:
			pass
	assert len(reorder_childs) == len(childs) - 1
	return reorder_childs
def reorder_sdrs(childs, tokens):
	reorder_childs = []
	for i in range(len(childs)-1):
		if kp_r.match(tokens[childs[i]+1]):
			reorder_childs.append((childs[i]+1, childs[i+1]))
		else:
			pass

	for i in range(len(childs)-1):
		if kp_r.match(tokens[childs[i]+1]):
			pass
		else:
			reorder_childs.append((childs[i]+1, childs[i+1]))

	
	assert len(reorder_childs) == len(childs) - 1
	return reorder_childs
def process(tokens):
	if len(tokens) == 2:
		return " ".join(tokens)
	elif len(tokens) == 3 and tokens[1][-1] != "(":
		return " ".join(tokens)
	elif len(tokens) == 4 and tokens[1][-1] != "(":
		return " ".join(tokens)

	childs = get_childs(tokens[1:-1])
	reorder_childs = reorder(childs, tokens)
	new_tokens = [tokens[0]]
	for start, end in reorder_childs:
		new_tokens.append(process(tokens[start:end+1]))
	new_tokens.append(tokens[-1])
	return " ".join(new_tokens)

def normal(tokens):
	x = []
	s = []
	e = []
	p = []
	for tok in tokens:
		if xp.match(tok):
			if not(tok in x):
				x.append(tok)
		elif sp.match(tok):
			if not(tok in s):
				s.append(tok)
		elif ep.match(tok):
			if not(tok in e):
				e.append(tok)
	new_tokens = []
	for tok in tokens:
		if xp.match(tok):
			idx = x.index(tok)
			new_tokens.append("X"+str(idx+1))
		elif sp.match(tok):
			idx = s.index(tok)
			new_tokens.append("S"+str(idx+1))
		elif ep.match(tok):
			idx = e.index(tok)
			new_tokens.append("E"+str(idx+1))
		else:
			new_tokens.append(tok)
	return " ".join(new_tokens)

for line in open(sys.argv[1]):
	line = line.strip()
	if line[0:4] == "DRS(":
		new_tokens = process(line.split())
		new_tokens = normal(new_tokens.split())
		print new_tokens
	else:
		print line

