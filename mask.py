

## ensure
## 1) bracket completed
## 2) relation(variables, variables) or relation(variables)

class SimpleMask:
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
				while idx < k_rel_start:
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


				


			
