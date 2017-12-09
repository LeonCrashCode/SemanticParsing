

class Mask:
	def __init__(self, tags):
		self.tag_size = len(tags.tag_to_ix)
		self.tags = tags

		self.reduce_ix = self.tags.tag_to_ix[self.reduce]
		self.sdrs_ix = self.tags.tag_to_ix[self.rel_sdrs]
		self.drs_ix = self.tags.tag_to_ix[self.rel_drs]
		self.not_ix = self.tags.tag_to_ix[self.rel_not]
		self.nec_ix = self.tags.tag_to_ix[self.rel_nec]
		self.pos_ix = self.tags.tag_to_ix[self.rel_pos]
		self.or_ix = self.tags.tag_to_ix[self.rel_or]
		self.duplex_ix = self.tags.tag_to_ix[self.rel_duplex]
		self.imp_ix = self.tags.tag_to_ix[self.rel_imp]
		self.gen_k_ix = self.tags.tag_to_ix[self.act_rel_k]

		self.SOS_mask = self._get_SOS_mask()
		self.sdrs_le1k_mask = self._get_sdrs_le1k_mask()
		self.sdrs_ge2k_mask = self._get_sdrs_ge2k_mask()
		self.sdrs_ge2k_nore_mask = self._get_sdrs_ge2k_nore_mask()
		self.sdrs_ge2k_re_mask = self._get_sdrs_ge2k_re_mask()

	def _get_SOS_mask(self):
		re = [0 for i in range(self.tag_size)]
		re[self.drs_ix] = 1
		return [ 0 , re]

	def _get_sdrs_le1k_mask(self):
		re = [0 for i in range(self.tag_size)]
		re[self.gen_k_ix] = 1
		return [ 0, re]

	def _get_sdrs_ge2k_nore_mask(self, ks):
		re = [0 for i in range(self.tag_size)]
		re[self.gen_k_ix] = 1
		return [ 0, re]

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
        	v_cnt = [0 for i in range(5)] # k p x e s
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
		re = [0 for i in range(self.tag_size)]

		if stack_tags[-1][0] == self.tags.SOS:
			re[self.drs_ix] = 1
			return [ 0 , re]
		elif stack_tags[-1][0] == self.tags.rel_sdrs:
			if stack_tags[-1][1] > 0:
				re[self.reduce_ix] = 1
			re[self.tags.k_tag_start+v_cnt[0]] = 1
			for i in range(len(self.tags.relation_global)):
				re[self.tags.global_start+i] = 1
		elif stack_tag[]

				


			
