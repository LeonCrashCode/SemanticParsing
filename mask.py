

class Mask:
	def __init__(self, tags):
		self.tag_size = len(tags.tag_to_ix)
		self.tags = tags

		self.sdrs = self._get_one_hot([tags.tag_sdrs])
		self.drs = self._get_one_hot([tags.tag_drs])
		self.special_tag = self._get_one_hot(tags.special_tag[2:])
		self.special_relation = self._get_one_hot(tags.special_relation)
		self.k_tag = self._get_one_hot(tags.K_tag)
		self.p_tag = self._get_one_hot(tags.P_tag)
		self.x_tag = self._get_one_hot(tags.X_tag)
		self.e_tag = self._get_one_hot(tags.E_tag)
		self.s_tag = self._get_one_hot(tags.S_tag)
		self.other_relations = self._get_one_hot(tags.relation_one_slot+tags.relation_two_slot+tags.relation_flexible_slot)
		self.reduce = self._get_one_hot([tags.reduce])

		self.open_bracket = self._get_one_hot()

	def _get_one_hot(self, hots):
		re = [0 for i in range(self.tag_size)]
		for hot in hots:
			re[self.tag_to_ix[hot]] = 1
		return re
	def get_mask_by_string(self, string, relation_cnt, open_bracket_cnt):
		return self.get_mask_by_idx(tags.tag_to_ix[string], relation_cnt, open_bracket_cnt)

	def get_mask_by_idx(self, idx, relation_cnt, open_bracket_cnt):
		if self.tags.isSOS(idx):
			return self.drs
		elif self.tags.issdrs(idx) or self.tags.isdrs(idx):
			return self.open_bracket
			
