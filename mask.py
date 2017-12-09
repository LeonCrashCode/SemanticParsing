

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
		re = [0 for i in range(self.tags.local_start)]
		for hot in hots:
			re[self.tag_to_ix[hot]] = 1
		return re

	def _isVar(self, tok):
		return (len(tok) >= 2 and (tok[0] in ['k', 'p', 'x', 'e', 's']) and tok[-1] != '(')
	def _get_Var_type(self, tok):
		return ['k', 'p', 's', 'x', 'e'].index(tok[0])

	def _isReduce(self, tok):
		return tok == self.tags.reduce

	def _isRelation(self, tok):
		return (tok[-1] == '(')
		
	def get_mask(self, trn_sentences):
    	trn_masks = []
    	for instance in trn_instances:
        	instance_masks = []

        	relation_cnt = 0
        	open_bracket = 0
        	v_cnt = [1 for i in range(5)] # k p x e s
        	stack_tags = [[self.tags.SOS, 0]]

        	instance_masks.append(self._get_mask(v_cnt, relation_cnt, open_bracket, stack_tags, len(instance[-1])))
        	for tok in instance[-1]:
        		if self._isVar(tok): # variables
        			stack_tags[-1][1] += 1
        			v_type = self._get_Var_type(tok)
        			v_cnt[v_type] = max(v_cnt[v_type], int(tok[1:]))

        		elif self._isReduce(tok):
        			stack_tags = stack_tags[:-1]
        			stack_tags[-1][1] += 1
        			open_bracket -= 1

        		elif self._isRelation(tok):
        			stack_tags.append([tok, 0])
        			relation_cnt += 1

            	instanec_masks.append(self._get_mask(v_cnt, relation_cnt, open_bracket, stack_tags))
            trn_masks.append(instance_masks)
        return trn_masks

	def _get_mask(self, v_cnt, relation_cnt, open_bracket, stack_tags, length):
		if stack_tags[-1][0] == self.tags.SOS:
			return [ [] , self._get_one_hot([self.tags.tag_drs])]
		elif stack_tags[-1][0] == self.tags.rel_sdrs:
			if stack_tags[-1][1] > 0:
				tmp = []
				tmp.append(self.tags.reduce)
				return [ [], self._get_one_hot([])]


			
