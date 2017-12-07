
class Tag:
	def __init__(self, filename):
		self.filename = filename

		self.SOS = "<SOS>"
		self.EOS = "<EOS>"
		self.reduce = ")"

		self.special_tag = ["sdrs(", "drs(", "not(", "nec(", "pos(", "or(", "duplex(", "imp("]
		self.tag_sdrs = "sdrs("
		self.tag_drs = "drs("
		self.tag_not = "not("
		self.tag_nec = "nec("
		self.tag_pos = "pos("
		self.tag_or = "or("
		self.tag_duplex = "duplex("
		self.tag_imp = "imp("

		self.special_relation = ["timex(", "card("]
		self.rel_timex = "timex("
		self.rel_card = "card("

		self.K_tag = list()
		self.P_tag = list()
		self.X_tag = list()
		self.E_tag = list()
		self.S_tag = list()
		self.relation_one_slot = list()
		self.relation_two_slot = list()
		self.relation_flexible_slot = list()
		self.relation_global = list()
		self.fr = open(filename,"r")

		while True:
			line = self.fr.readline().strip()
			if line == "":
				break
			if len(line.split()) == 3 and line.split()[0] == "====":
				if line.split()[1] == "K_tag":
					self.K_tag = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "P_tag":
					self.P_tag = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "X_tag":
					self.X_tag = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "E_tag":
					self.E_tag = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "S_tag":
					self.S_tag = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "relation_one_slot":
					self.relation_one_slot = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "relation_two_slot":
					self.relation_two_slot = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "relation_flexible_slot":
					self.relation_flexible_slot = self._readblock(int(line.split()[2]))
				elif line.split()[1] == "relation_global":
					self.relation_global = self._readblock(int(line.split()[2]))
				else:
					assert False, "unrecogized type"
		self.fr.close()

		self.tag_to_ix = {SOS:0, EOS:1, self.reduce:2}
		self.ix_to_tag = ["<SOS>", "<EOS>", self.reduce]

		for tag in self.special_tag:
			self.tag_to_ix[tag] = len(self.tag_to_ix)
			self.ix_to_tag.append(tag)

		for tag in self.special_relation:
			self.tag_to_ix[tag] = len(self.tag_to_ix)
			self.ix_to_tag.append(tag)

		for tag in self.K_tag:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)
    	for tag in self.P_tag:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)
    	for tag in self.X_tag:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)
    	for tag in self.E_tag:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)
    	for tag in self.S_tag:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)

    	for tag in self.relation_one_slot:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)
    	for tag in self.relation_two_slot:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)
    	for tag in self.relation_flexible_slot:
    		self.tag_to_ix[tag] = len(self.tag_to_ix)
    		self.ix_to_tag.append(tag)

	def _readblock(self, blocksize):
		temp = []
		for i in range(blocksize):
			temp.append(self.fr.readline().strip())
		return temp
	def print_info(self):
		print "special_tag", len(self.special_tag), " ".join(self.special_tag)
		print "special_relation", len(self.special_relation), " ".join(self.special_relation)
		print "K_tag", len(self.K_tag), " ".join(self.K_tag)
		print "P_tag", len(self.P_tag), " ".join(self.P_tag)
		print "X_tag", len(self.X_tag), " ".join(self.X_tag)
		print "E_tag", len(self.E_tag), " ".join(self.E_tag)
		print "S_tag", len(self.S_tag), " ".join(self.S_tag)
		print "relation", len(self.relation_one_slot)+len(self.relation_two_slot)+len(self.relation_flexible_slot)



