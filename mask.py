

## ensure
## 1) bracket completed
## 2) relation(variables, variables) or relation(variables)

class SimpleMask:
	def __init__(self, tags_info, encoder_input_size, inputs, conditions={}):
		
		relation_count = 0
		stack = []

		if len(conditions) !=0:
			relation_count = conditions["relation_count"]
			stack = conditions["stack"]
		inputs = [[-2, 0]] + inputs
		res = []
		for type, ix in inputs:
			if len(res) != 0:
				if type == -2:
					assert res[-1][ix] != 0
				else:
					assert res[-1][type+tags_info.tag_size] != 0
			####
			if len(stack) == 0:
				stack.append(999)
			elif ix < tags_info.tag_size:
				if ix >= 5 and ix <=12:
					stack.append(0)
					relation_count += 1
				elif ix > 12 and ix < tags_info.k_rel_start:
					stack.append(1)
					relation_count += 1
				elif ix >= tags_info.k_rel_start and ix < tags_info.k_tag_start:
					stack.append(0)
					relation_count += 1
				elif ix == 4:
					while stack[-1] < 0:
						stack.pop()
					stack[-1] = -2
				else:
					stack.append(-1)
			else:
				stack.append(1)
				relation_count += 1

			if stack[-1] == 999:
				re = self._get_zeros(tags_info.tag_size) + self._get_zeros(encoder_input_size)
				re[tags_info.tag_to_ix[tags_info.rel_drs]] = 1
				res.append(re)
			elif stack[-1] == 0:
				if relation_count > 200:
					re = self._get_zeros(tags_info.tag_size) + self._get_ones(encoder_input_size)
					re[tags_info.tag_to_ix[tags_info.rel_timex]] = 1
					re[tags_info.tag_to_ix[tags_info.rel_card]] = 1
					idx = tags_info.global_start
					while idx < k_rel_start:
						re[idx] = 1
						idx += 1
					res.append(re)
				else:
					re = self._get_ones(tags_info.tag_size) + self._get_ones(encoder_input_size)
					re[0] = 0
					re[1] = 0
					re[2] = 0
					re[3] = 0
					idx = tags_info.k_tag_start
					while idx < tags_info.tag_size:
						re[idx] = 0
						idx += 1
					res.append(re)
			elif stack[-1] == 1:
				re = self._get_zeros(tags_info.tag_size) + self._get_zeros(encoder_input_size)
				re[2] = 1
				re[3] = 1
				idx = tags_info.k_tag_start
				while idx < tags_info.tag_size:
					re[idx] = 1
					idx += 1
				res.append(re)
			elif stack[-1] == -1:
				assert len(stack) >= 3
				if stack[-2] == -1:
					re = self._get_zeros(tags_info.tag_size) + self._get_zeros(encoder_input_size)
					re[tags_info.tag_to_ix[tags_info.reduce]] = 1
					res.append(re)
				else:
					re = self._get_zeros(tags_info.tag_size) + self._get_zeros(encoder_input_size)
					re[tags_info.tag_to_ix[tags_info.reduce]] = 1
					re[2] = 1
					re[3] = 1
					idx = tags_info.k_tag_start
					while idx < tags_info.tag_size:
						re[idx] = 1
						idx += 1
					res.append(re)
			elif stack[-1] == -2:
				if stack[-2] == 999:
					re = self._get_zeros(tags_info.tag_size) + self._get_zeros(encoder_input_size)
					re[tags_info.tag_to_ix[tags_info.EOS]] = 1
					res.append(re)
				elif relation_count > 200:
					re = self._get_zeros(tags_info.tag_size) + self._get_zeros(encoder_input_size)
					re[tags_info.tag_to_ix[tags_info.reduce]] = 1
					res.append(re)
				else:
					re = self._get_ones(tags_info.tag_size) + self._get_ones(encoder_input_size)
					re[0] = 0
					re[1] = 0
					re[2] = 0
					re[3] = 0
					idx = tags_info.k_tag_start
					while idx < tags_info.tag_size:
						re[idx] = 0
						idx += 1
					res.append(re)
			else:
				assert False
			

	def _get_zeros(self, size):
		return [0 for i in range(size)]

	def _get_ones(self, size):
		return [1 for i in range(size)]


				


			
