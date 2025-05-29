from torch.nn import Module, Embedding, Tanh, Sequential, Softmax, \
    Sigmoid, Parameter, Linear, Conv1d, ModuleList, ELU
import torch.nn as nn
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer
import torch as T

class CustomBartLayer(nn.Module):
    def __init__(self, z_fun, bart_layer):
        super().__init__()
        self.bart_layer = bart_layer
        self.project_down = Linear(768, 384)
        self.project_up = Linear(384, 768)
        self.z_fun = z_fun
        self.p_q = dict()

    def forward(self, *x, layer_head_mask=None, output_attentions=None):
        bert_out = self.bart_layer(*x, layer_head_mask=None)
        down_x = self.project_down(bert_out[0])
        down_x = T.transpose(down_x, 2, 1)
        latent_x, all_q, all_p, all_log_q, all_log_p = self.z_fun(s=down_x)
        up_x = self.project_up(latent_x)
        return (up_x,)


class MainModel(Module):

    def __init__(self,):

        super(MainModel, self).__init__()
        self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', )
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.bart_model.model.encoder.layers[1] = CustomBartLayer(self.get_z, self.bart_model.model.encoder.layers[1])

    def forward(self, rev):


        # rev is encoder_input_ids
        output = self.bart_model(rev)
        fianl_output = output.last_hidden_state
