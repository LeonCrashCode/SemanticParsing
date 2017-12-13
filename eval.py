import re
class Eval:
	def __init__(self, tags_info):
		self.ignore = ["SDRS(", "DRS(", "NOT(", "NEC(", "POS(", "OR(", "DUPLEX(", "IMP(", ")"]
		
		self.kp = re.compile("^K[0-9]+\(?$")
		self.pp = re.compile("^K[0-9]+\(?$")
		self.xp = re.compile("^K[0-9]+\(?$")
		self.ep = re.compile("^K[0-9]+\(?$")
		self.sp = re.compile("^K[0-9]+\(?$")

	def eval(self, outputs, golds):
		assert len(outputs) == len(golds)

		for i in range(len(outputs)):
			self._eval(outputs[i], golds[i])

	def _eval(self, output, gold):
		o = self._get_relation_set(output)
		g = self._get_relation_set(gold)

		p_base = len(o)
		r_base = len(g)

		acc = 0.0
		for item in o:
			if item in g:
				acc += 1

		P = acc/p_base
		R = acc/r_base

		F = 0
		if acc != 0:
			F = 2*P*R/(P+R)
		return P,R,F

	def _get_relation_set(self, drs):
		drs = drs.split()
		re = []
		for item in drs:
			if item in self.ignore:
				pass
			elif self.kp.match(item):
				pass
			elif self.pp.match(item):
				pass
			elif self.xp.match(item):
				pass
			elif self.ep.match(item):
				pass
			elif self.sp.match(item):
				pass
			else:
				re.append(item)


