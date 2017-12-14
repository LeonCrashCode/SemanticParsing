import sys
from sets import Set
import re

kp_r = re.compile("^K([0-9]+)\($")
pp_r = re.compile("^P([0-9]+)\($")

xp = re.compile("^X([0-9]+)$")
ep = re.compile("^E([0-9]+)$")
sp = re.compile("^S([0-9]+)$")
tp = re.compile("^T([0-9]+)$")
kp = re.compile("^K([0-9]+)$")


drs = ["DRS(", "SDRS(", "NOT(", "POS(", "NEC(", "OR(", "IMP(", "DUPLEX("]
def is_variables(tok):
	if xp.match(tok) or ep.match(tok) or sp.match(tok) or tp.match(tok) or kp.match(tok):
		return True
	return False

def process(tokens):
	current_b = 0
	variables = Set()
	for i in tokens:
		if tok in drs:
			stack.append("b"+str(current_b))
			current_b += 1
		elif kp_r.match(tok) or pp_r.match(tok):
			stack.append(tok)
			if tok[-1] not in variables:
				tuples.append([stack[-1], "REF", tok[-1]])
			variables.add(tok[-1])
		elif is_variables(tok):
			if tok not in variables:
				tuples.append([stack[-1], "REF", tok])
			variables.add(tok)
		elif 
		elif tok == ")":
			stack.pop()
		elif 



for line in open(sys.argv[1]):
	line = line.strip()
	if line[:4] == "DRS(":
		process(line.split())