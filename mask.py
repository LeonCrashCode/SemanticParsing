


class SimpleMask:
	## 1) bracket completed, ensuring left bracket "(" and right bracket ")" appear pairly.
	## 2) relation(variables, variables) or relation(variables)
	def __init__(self, tags_info, encoder_input_size=0):
		self.tags_info = tags_info
		self.reset(encoder_input_size)
		self.mask = 0
		self.need = 1
	def reset(self, encoder_input_size):
		self.relation_count = 0
		self.stack = [999]
		self.encoder_input_size = encoder_input_size

	def get_all_mask(self, inputs):
		res = []
		res.append(self.get_step_mask())
		for type, ix in inputs:
			if type == -2:
				assert res[-1][ix] != self.mask
			else:
				assert res[-1][type+self.tags_info.tag_size] != self.mask
			self.update(type, ix)
			res.append(self.get_step_mask())
		return res

	def get_step_mask(self):
		if self.stack[-1] == 999:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[self.tags_info.tag_to_ix[self.tags_info.rel_drs]] = self.need
			return re
		elif self.stack[-1] == 0:
			if self.relation_count > 200:
				re = self._get_zeros(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
				re[self.tags_info.tag_to_ix[self.tags_info.rel_timex]] = self.need
				re[self.tags_info.tag_to_ix[self.tags_info.rel_card]] = self.need
				re[self.tags_info.tag_to_ix[self.tags_info.rel_eq]] = self.need
				idx = self.tags_info.global_start
				while idx < self.tags_info.k_rel_start:
					re[idx] = self.need
					idx += 1
				return re
			else:
				re = self._get_ones(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
				re[0] = self.mask
				re[1] = self.mask
				re[2] = self.mask
				re[3] = self.mask
				idx = self.tags_info.k_tag_start
				while idx < self.tags_info.tag_size:
					re[idx] = self.mask
					idx += 1
				return re
		elif self.stack[-1] == 1:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[2] = self.need
			re[3] = self.need
			idx = self.tags_info.k_tag_start
			while idx < self.tags_info.tag_size:
				re[idx] = self.need
				idx += 1
			return re
		elif self.stack[-1] == -1:
			assert len(self.stack) >= 3
			if self.stack[-2] == -1:
				re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
				re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
				return re
			else:
				re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
				re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
				re[2] = self.need
				re[3] = self.need
				idx = self.tags_info.k_tag_start
				while idx < self.tags_info.tag_size:
					re[idx] = self.need
					idx += 1
				return re
		elif self.stack[-1] == -2:
			if self.stack[-2] == 999:
				re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
				re[self.tags_info.tag_to_ix[self.tags_info.EOS]] = self.need
				return re
			elif self.relation_count > 200:
				re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
				re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
				return re
			else:
				re = self._get_ones(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
				re[0] = self.mask
				re[1] = self.mask
				re[2] = self.mask
				re[3] = self.mask
				idx = self.tags_info.k_tag_start
				while idx < self.tags_info.tag_size:
					re[idx] = self.mask
					idx += 1
				return re
		else:
			assert False
		
	def update(self, type, ix):
		if ix < self.tags_info.tag_size:
			if ix >= 5 and ix <=12:
				self.stack.append(0)
				self.relation_count += 1
			elif ix > 12 and ix < self.tags_info.k_rel_start:
				self.stack.append(1)
				self.relation_count += 1
			elif ix >= self.tags_info.k_rel_start and ix < self.tags_info.k_tag_start:
				self.stack.append(0)
				self.relation_count += 1
			elif ix == 4:
				while self.stack[-1] < 0:
					self.stack.pop()
				self.stack[-1] = -2
			else:
				self.stack.append(-1)
		else:
			self.stack.append(1)
			self.relation_count += 1
	def _print_state(self):
		print "relation_count", self.relation_count
		print "stack", self.stack
	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]


class StructuredMask:
	#sdrs should have at least two k(), at least one relation, and the relation should follow k()
	#drs should have at least anything, except variables.
	#not, nec, pos should have and only have one drs or sdrs
	#imp, or, duplex should have and only have two drs or sdrs
	#timex should be timex(variables, TIME_NUMBER)
	#card should be card(variables, CARD_NUMBER)
	#k(, p( should have and only have one drs or sdrs
	#variables index constraints
	def __init__(self, tags_info, encoder_input_size=0):
		self.tags_info = tags_info
		self.mask = 0
		self.need = 1

		self.SOS = tags_info.all_tag_size*10
		self.relation = tags_info.all_tag_size*10+1

		self.variable_offset = 0
		self.relation_offset = 1
		self.k_relation_offset = 2
		self.p_relation_offset = 3
		self.drs_offset = 4
		self.six_offset = 5
		
		self.reset(encoder_input_size)
	def reset(self, encoder_input_size):
		self.relation_count = 0
		self.stack = [self.SOS]
		self.encoder_input_size = encoder_input_size
		self.stack_ex = [[0 for i in range(6)]]
		self.k = 1
		self.p = 1
		self.x = 1
		self.e = 1
		self.s = 1

	def get_all_mask(self, inputs):
		res = []
		res.append(self.get_step_mask())
		for type, ix in inputs:
			#self._print_state()
			#print res[-1]
			if type == -2:
				assert res[-1][ix] != self.mask
			else:
				assert res[-1][type+self.tags_info.tag_size] != self.mask
			self.update(type, ix)
			res.append(self.get_step_mask())
		return res

	def get_step_mask(self):
		if self.stack[-1] == self.SOS:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[self.tags_info.tag_to_ix[self.tags_info.rel_drs]] = self.need
			return re
		elif self.stack[-1] == 5:
			#SDRS
			return self._get_sdrs_mask()
		elif self.stack[-1] == 6:
			#DRS
			return self._get_drs_mask()
		elif self.stack[-1] in [7, 8, 9]:
			#not, nec, pos
			return self._get_1_mask()
		elif self.stack[-1] in [10, 11, 12]:
			#or, imp, duplex
			return self._get_2_mask()
		elif self.stack[-1] == 13:
			#timex
			return self._get_timex_mask()
		elif self.stack[-1] == 14:
			#card
			return self._get_card_mask()
		elif self.stack[-1] == 15 or self.stack[-1] == self.relation:
			#relatoin
			return self._get_relation_mask()
		elif self.stack[-1] == self.tags_info.k_rel_start or self.stack[-1] == self.tags_info.p_rel_start:
			#k p
			return self._get_1_mask()
		else:
			assert False
	def _get_sdrs_mask(self):
		#SDRS
		if self.stack_ex[-1][self.k_relation_offset] < 2:
			#only k
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			idx = self.tags_info.k_rel_start
			while idx < self.tags_info.p_rel_start:
				re[idx] = self.need
				idx += 1
			return re
		elif self.stack_ex[-1][self.relation_offset] == 0:
			#only relation
			re = self._get_zeros(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
			idx = self.tags_info.global_start
			while idx < self.tags_info.k_rel_start:
				re[idx] = self.need
				idx += 1
			if self.relation_count <= 200:
				# k is ok
				idx = self.tags_info.k_rel_start
				while idx < self.tags_info.p_rel_start:
					re[idx] = self.need
					idx += 1
			return re
		else:
			#only reduce
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
			if self.relation_count <= 200:
				idx = self.tags_info.global_start
				while idx < self.tags_info.k_rel_start:
					re[idx] = self.need
					idx += 1
				idx = self.tags_info.tag_size
				while idx < len(re):
					re[idx] = self.need
					idx += 1
			return re
	def _get_drs_mask(self):
		if (self.stack_ex[-1][self.relation_offset] + self.stack_ex[-1][self.six_offset] + self.stack_ex[-1][self.p_relation_offset]) == 0:
			re = self._get_ones(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
			re[0] = self.mask #SOS
			re[1] = self.mask #EOS
			re[2] = self.mask #TIME_NUMBER
			re[3] = self.mask #CARD_NUMBER
			re[4] = self.mask #reduce
			re[5] = self.mask #sdrs
			re[6] = self.mask #drs
			idx = self.tags_info.k_tag_start
			while idx < self.tags_info.tag_size:
				re[idx] = self.mask
				idx += 1
			if self.relation_count > 200:
				re[7] = self.mask
				re[8] = self.mask
				re[9] = self.mask
				re[10] = self.mask
				re[11] = self.mask
				re[12] = self.mask
			return re
		else:
			if self.relation_count <= 200:
				re = self._get_ones(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
				re[0] = self.mask #SOS
				re[1] = self.mask #EOS
				re[2] = self.mask #TIME_NUMBER
				re[3] = self.mask #CARD_NUMBER
				re[5] = self.mask #sdrs
				re[6] = self.mask #drs
				idx = self.tags_info.k_tag_start
				while idx < self.tags_info.tag_size:
					re[idx] = self.mask
					idx += 1
				return re
			else:
				re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
				re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
				return re
	def _get_1_mask(self):
		if self.stack_ex[-1][self.drs_offset] == 0:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[5] = self.need
			re[6] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[4] = self.need
			return re
	def _get_2_mask(self):
		if self.stack_ex[-1][self.drs_offset] <= 1:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[5] = self.need
			re[6] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[4] = self.need
			return re
		
	def _get_timex_mask(self):
		if self.stack_ex[-1][self.variable_offset] == 0:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			idx = self.tags_info.k_tag_start
			while idx < self.tags_info.tag_size:
				re[idx] = self.need
				idx += 1
			return re
		elif self.stack_ex[-1][self.variable_offset] == 1:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[3] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[4] = self.need
			return re
	def _get_card_mask(self):
		if self.stack_ex[-1][self.variable_offset] == 0:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			idx = self.tags_info.k_tag_start
			while idx < self.tags_info.tag_size:
				re[idx] = self.need
				idx += 1
			return re
		elif self.stack_ex[-1][self.variable_offset] == 1:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[2] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[4] = self.need
			return re
	def _get_relation_mask(self):
		if self.stack_ex[-1][self.variable_offset] <= 1:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			idx = self.tags_info.k_tag_start
			while idx < self.tags_info.tag_size:
				re[idx] = self.need
				idx += 1
			if self.stack_ex[-1][self.variable_offset] == 1:
				re[4] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size) + self._get_zeros(self.encoder_input_size)
			re[4] = self.need
			return re

	def update(self, type, ix):
		if ix < self.tags_info.tag_size:
			if ix >= 5 and ix < self.tags_info.global_start:
				self.stack.append(ix)
				self.relation_count += 1
				self.stack_ex.append([0 for i in range(6)])
			elif ix >= self.tags_info.global_start and ix < self.tags_info.k_rel_start:
				self.stack.append(self.relation)
				self.relation_count += 1
				self.stack_ex.append([0 for i in range(6)])
			elif ix >= self.tags_info.k_rel_start and ix < self.tags_info.p_rel_start:
				self.stack.append(self.tags_info.k_rel_start)
				self.relation_count += 1
				self.stack_ex.append([0 for i in range(6)])
			elif ix >= self.tags_info.p_rel_start and ix < self.tags_info.k_tag_start:
				self.stack.append(self.tags_info.p_rel_start)
				self.relation_count += 1
				self.stack_ex.append([0 for i in range(6)])
			elif ix == 4:
				self.stack_ex.pop()
				if self.stack[-1] == 5 or self.stack[-1] == 6:
					self.stack_ex[-1][self.drs_offset] += 1
				elif self.stack[-1] >= 7 and self.stack[-1] <= 12:
					self.stack_ex[-1][self.six_offset] += 1
				elif self.stack[-1] >= 13 and self.stack[-1] <= 15:
					self.stack_ex[-1][self.relation_offset] += 1
				elif self.stack[-1] == self.relation:
					self.stack_ex[-1][self.relation_offset] += 1
				elif self.stack[-1] == self.tags_info.k_rel_start:
					self.stack_ex[-1][self.k_relation_offset] += 1
				elif self.stack[-1] == self.tags_info.p_rel_start:
					self.stack_ex[-1][self.p_relation_offset] += 1
				else:
					assert False
				self.stack.pop()
			else:
				self.stack_ex[-1][self.variable_offset] += 1
		else:
			self.stack.append(self.relation)
			self.relation_count += 1
			self.stack_ex.append([0 for i in range(6)])


	def _print_state(self):
		print "relation_count", self.relation_count
		print "stack", self.stack
		print "stack_ex", self.stack_ex
	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]

