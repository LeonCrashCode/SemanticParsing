import sys
from sets import Set
import re

kp_r = re.compile("^K([0-9]+)\($")
pp_r = re.compile("^P([0-9]+)\($")

xp = re.compile("^X([0-9]+)$")
ep = re.compile("^E([0-9]+)$")
sp = re.compile("^S([0-9]+)$")
pp = re.compile("^P([0-9]+)$")
kp = re.compile("^K([0-9]+)$")


special = ["NOT(", "POS(", "NEC(", "OR(", "IMP(", "DUPLEX("]

def is_variables(tok):
	if xp.match(tok) or ep.match(tok) or sp.match(tok) or pp.match(tok) or kp.match(tok):
		return True
	return False

def get_b(stack):
	for item in stack[::-1]:
		if item[0] == "b":
			return item

def process(tokens):
	current_b = 0
	current_c = 0
	variables = Set()
	i = 0
	stack = []
	tuples = []
	print tokens
	while i < len(tokens):
		tok = tokens[i]
		print stack
		print tuples
		if tok == "<EOS>":
			i += 1
			pass
		elif tok == "DRS(" or tok == "SDRS(":
			stack.append("b"+str(current_b))
			current_b += 1
			i+= 1
		elif tok in special:
			stack.append("c-"+str(current_c)+"-"+str(special.index(tok)))
			current_c += 1
			i+=1
		elif kp_r.match(tok) or pp_r.match(tok):
			stack.append(tok[:-1])
			if tok[-1] not in variables:
				tuples.append([stack[-1], "REF", tok[:-1]])
				variables.add(tok[:-1])
			i += 1
		elif tok == "TIMEX(" or tok == "CARD(":
			if tokens[i+1] not in variables:
				tuples.append([get_b(stack), "REF", tokens[i+1]])
				variables.add(tokens[i+1])
			tuples.append([get_b(stack), "TIMEX", "c"+str(current_c)])
			tuples.append(["c"+str(current_c), "ARG1", tokens[i+1]])
			if tokens[i+2] != ")":
				tuples.append(["c"+str(current_c), "ARG2", '"'+tokens[i+2]+'"'])
			current_c += 1

			if tokens[i+2] != ")":
				i += 4
			else:
				i += 3
		elif tok == ")":
			tmp = []
			while stack[-1][0] == "-":
				tmp.append(stack.pop())
			tmp = tmp[::-1]

			if stack[-1][0] == "c":
				assert len(tmp) <= 2
				crel = special[int(stack[-1].split("-")[2])]
				cnum = " ".join(stack[-1].split("-")[:2])
				tuples.append([get_b(stack), crel, cnum])
				tuples.append([cnum, "ARG1", tmp[0][1:]])
				if len(tmp) == 2:
					tuples.append([cnum, "ARG1", tmp[0][1:]])
				stack.pop()
			elif stack[-1][0] in ["k","p"]:
				assert len(tmp) == 1
				if stack[-1][0] == "k":
					tuples.append([get_b(stack), "CONSTITUENT", str(current_c)])
				else:
					tuples.append([get_b(stack), "PROP", str(current_c)])
				
				tuples.append([str(current_c), "ARG1", stack[-1]])
				tuples.append([str(current_c), "ARG2", tmp[0][1:]])
				current_c += 1
			else:
				stack[-1] = "-"+stack[-1]
			i += 1
		else:
			if tokens[i+1] not in variables:
				tuples.append([get_b(stack), "REF", tokens[i+1]])
				variables.add(tokens[i+1])
			tuples.append([get_b(stack), tok, "c"+str(current_c)])
			tuples.append(["c"+str(current_c), "ARG1", tokens[i+1]])
			if tokens[i+2] != ")":
				if tokens[i+2] not in variables:
					tuples.append([get_b(stack), "REF", tokens[i+2]])
					variables.add(tokens[i+2])
				tuples.append(["c"+str(current_c), "ARG2", tokens[i+2]])
			current_c += 1
			if tokens[i+2] != ")":
				i += 4
			else:
				i += 3

for line in open(sys.argv[1]):
	line = line.strip()
	if line[:4] == "DRS(":
		process(line.split())
