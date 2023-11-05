class ModelF(object):
    # core model
    REV = 'rev'
    REV_LEN = 'rev_len'
    REV_MASK = 'rev_mask'
    PROMPT = 'prompt'
    PROMPT_LEN = 'prompt_len'
    PROMPT_MASK = 'prompt_mask'
    GROUP_REV_INDXS = 'group_rev_indxs'
    GROUP_REV_INDXS_MASK = 'group_rev_indxs_mask'
    SUMM_GROUP_ID = 'summ_group_id'
    SUMM_CAT = 'summ_category'

    # attention + copy mechanism
    REV_TO_GROUP_INDX = 'rev_to_group_indx'
    OTHER_REV_INDXS = 'other_rev_indxs'
    OTHER_REV_INDXS_MASK = 'other_rev_indxs_mask'
    OTHER_REV_COMP_STATES = 'other_rev_comp_states'
    OTHER_REV_COMP_STATES_MASK = 'other_rev_comp_states_mask'

    # interfaces
    GROUP_ID = 'group_id'
    CAT = 'category'

    GEN_SUMM = 'gen_summ'
    GEN_REV = 'gen_rev'
    GEN_Z1 = 'gen_z1'
    GEN_Z2 = 'gen_z2'
    GEN_Z3 = 'gen_z3'
    GEN_Z4 = 'gen_z4'

    # for summary eval
    ROUGE = 'rouge'

    # for extra analysis
    COPY_ANALYSIS = 'copy_analysis'

    SUMMS = 'summs'
