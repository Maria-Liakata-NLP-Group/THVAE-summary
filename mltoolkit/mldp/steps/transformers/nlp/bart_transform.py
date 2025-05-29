from mltoolkit.mldp.steps.transformers.base_transformer import BaseTransformer
from transformers import BartTokenizer
from operator import itemgetter
import numpy as np
import torch

class BartTransform(BaseTransformer):
    """
    transform batch  sentences to id
    it inlucdes padding
    """
    def __init__(self, field_names_to_vocabs, bart_tokenizer, **kwargs):
        """
        :param field_names_to_vocabs: list of mappings to vocab objects.
        """
        super(BartTransform, self).__init__(**kwargs)
        self.field_names_to_vocabs = field_names_to_vocabs
        self.tokenizer = bart_tokenizer

    def _transform(self, data_chunk):
        keys = ['input_ids', 'attention_mask']
        for fn in self.field_names_to_vocabs:
            # data_chunk[fn], data_chunk['rev_mask'] = self.map_rev(data_chunk[fn], keys)
            data_chunk[fn] = self.map_rev(data_chunk[fn], keys)
        return data_chunk

    def map_rev(self, fv, keys):
        fv_str = list()
        for sen_list in fv:
            sen_str = ' '.join(sen_list)
            fv_str.append(sen_str)

        output = self.tokenizer(fv_str, padding='longest', truncation=True)
        sen_id, rev_mask = itemgetter(*keys)(output)
        # rev_mask = np.array(rev_mask)
        sen_id = np.array(sen_id)
        return sen_id



