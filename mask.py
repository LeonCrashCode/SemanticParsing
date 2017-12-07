

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
	def _isKVar(self, idx):
		return (self.tags.ix_to_tag[idx] in self.tags.K_tag)
	def _isPVar(self, idx):
		return (self.tags.ix_to_tag[idx] in self.tags.P_tag)
	def _isXVar(self, idx):
		return (self.tags.ix_to_tag[idx] in self.tags.X_tag)
	def _isEVar(self, idx):
		return (self.tags.ix_to_tag[idx] in self.tags.E_tag)
	def _isSVar(self, idx):
		return (self.tags.ix_to_tag[idx] in self.tags.S_tag)
	def _isVar(self, idx):
		return (self._isPVar(idx) or self._isKVar(idx) or self._isXVar(idx) or self.is_EVar(idx) or self.is_SVar(idx))
	
	def _isReduce(self, idx):
		return (self.tags.ix_to_tag[idx] == self.tags.reduces)

	def _isRelation(self, idx):
		return (self.tags.ix_to_tag[idx][-1] == '(')
		
	def get_mask(self, trn_sentences):
    	trn_masks = []
    	for instance in trn_instances:
        	instance_masks = []

        	relation_cnt = 0
        	open_bracket = 0
        	stack_tags = [[self.tags.SOS,0]]

        	instance_masks.append(self._get_mask(relation_cnt, open_bracket, stack_tags))
        	target_side = instance[-1].view(-1).data.tolist()
        	for idx in target_side:
        		if self._isVar(idx)
        			stack_tags[-1][1] += 1
        		elif self._isReduce(idx):
        			stack_tags = stack_tags[:-1]
        			stack_tags[-1][1] += 1
        			open_bracket -= 1
        		elif self._isRelation(idx):
        			stack_tags.append([self.tags.ix_to_tag[idx], 0])
        			relation_cnt += 1
            	instanec_masks.append(self._get_mask(relation_cnt, open_bracket, stack_tags))
            trn_masks.append(instance_masks)
        return trn_masks

	def _get_mask(self, relation_cnt, open_bracket, stack_tags)
			
