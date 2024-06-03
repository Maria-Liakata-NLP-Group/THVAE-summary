import torch
from mltoolkit.mlmo.layers import Ffnn, MuSigmaFfnn, Attention, OutEmbds
from mltoolkit.mlmo.layers.encoders import GruEncoder
from mltoolkit.mlmo.layers.decoders import PointerGenNetwork, GruPointerDecoder
from torch.nn import Module, Embedding, Tanh, Sequential, Softmax, \
    Sigmoid, Parameter, Linear, Conv1d, ModuleList, ELU
from mltoolkit.mlmo.utils.helpers.pytorch.computation import comp_seq_log_prob, \
    re_parameterize, kld_gauss, kld_normal, masked_softmax
from collections import OrderedDict
import torch as T
from thvae.utils.helpers.modelling import group_att_over_input, group_att_over_input_rev
from mltoolkit.mlmo.utils.tools import DecState
from distributions import Normal, DiscMixLogistic, NormalDecoder
from utils import get_stride_for_cell_type, get_input_size, groups_per_scale
from neural_operations import OPS, EncCombinerCell, DecCombinerCell, Conv1D, get_skip_connection, SE
from neural_ar_operations import ARConv1d, ARInvertedResidual, MixLogCDFParam, mix_log_cdf_flow
from neural_ar_operations import ELUConv as ARELUConv
import torch.nn.functional as F
import torch.nn as nn
from thirdparty.inplaced_sync_batchnorm import SyncBatchNormSwish
# from pre_train_transformer import load_parameter
from transformer.Models import Decoder, Encoder
import signatory
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, BartForCausalLM
from transformers.models.bart.modeling_bart import shift_tokens_right
# from peft import LoraConfig
# from transformers import AdapterLayer
# from esig import tosig as ts
# import iisignature
# from .siglayer import Signature

EPS = 1e-12

CHANNEL_MULT = 2

class Cell(Module):
    def __init__(self, Cin, Cout, cell_type, arch, use_se):
        super(Cell, self).__init__()
        self.cell_type = cell_type

        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, affine=False, channel_mult=CHANNEL_MULT)
        self.use_se = use_se
        self._num_nodes = len(arch)
        self._ops = ModuleList()
        # add different layer and convolutional layer
        # 2 or 1 nodes
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            C = Cin if i == 0 else Cout
            primitive = arch[i]
            op = OPS[primitive](C, Cout, stride)
            self._ops.append(op)

        if self.use_se:
            self.se = SE(Cout, Cout)

    def forward(self, s):
        # 看到没，返回的还是一个直接输出和一系列操作的和
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)

        s = self.se(s) if self.use_se else s

        return skip + 0.1 * s
        # return skip + s

class CellAR(Module):
    def __init__(self,num_z, num_ftr, num_c, arch, mirror):
        super(CellAR, self).__init__()
        assert num_c % num_z == 0

        self.cell_type = 'ar_nn'

        # s0 will the random samples
        ex = 6
        self.conv = ARInvertedResidual(num_z, num_ftr, ex=ex, mirror=mirror)
        # 0.1 helps bring mu closer to 0 initially
        self.mu = ARELUConv(self.conv.hidden_dim, num_z, kernel_size=1, padding=0, zero_diag=False,
                            weight_init_coeff=0.1, mirror=mirror)
    def forward(self, z, ftr):
        s = self.conv(z, ftr)

        mu = self.mu(s)
        new_z = (z-mu)
        log_det = torch.zeros_like(new_z)

        return new_z, log_det

class PairedCellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch=None):
        super(PairedCellAR, self).__init__()
        self.cell1 = CellAR(num_z, num_ftr, num_c, arch, mirror=False)
        self.cell2 = CellAR(num_z, num_ftr, num_c, arch, mirror=True)

    def forward(self, z, ftr):
        new_z, log_det1 = self.cell1(z, ftr)
        new_z, log_det2 = self.cell2(new_z, ftr)

        log_det1 += log_det2
        return new_z, log_det1


class CustomBartLayer(nn.Module):
    def __init__(self, z_fun, bart_layer):
        super().__init__()
        self.bart_layer = bart_layer
        self.adapter_norm_before = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.2)
        self.non_linearity = ELU()
        self.adapter_gate = Linear(768, 768)
        T.nn.init.xavier_uniform_(self.adapter_gate.weight)
        if self.adapter_gate.bias is not None:
            T.nn.init.zeros_(self.adapter_gate.bias)

        self.adapter_norm_after = nn.LayerNorm(768)

        self.z_fun = z_fun
        self.p_q = dict()

        self.down_normal = None

    def forward(self, *x, layer_head_mask=None, output_attentions=None):
        bert_out = self.bart_layer(*x, layer_head_mask=None)
        self.down_normal = self.adapter_norm_before(bert_out[0])
        # down_x = self.adapter_down(self.down_normal)
        down_x = self.dropout(self.down_normal)
        down_x = self.adapter_gate(down_x)
        down_x = self.dropout(down_x)
        down_x = self.non_linearity(down_x)
        down_x = T.transpose(down_x, 2, 1)
        latent_x, all_q, all_p, all_log_q, all_log_p = self.z_fun(s=down_x)
        self.p_q['all_q'] = all_q
        self.p_q['all_p'] = all_p
        self.p_q['all_log_q'] = all_log_q
        self.p_q['all_log_p'] = all_log_p
        # up_x = self.adapter_up(latent_x)
        # output = self.adapter_norm_after(up_x)
        output = self.adapter_norm_after(latent_x)
        return (output, )

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = nn.ReLU()

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores


class ThVAE(Module):
    """
    CopyCat summarizer variational summarizer based on hierarchical latent
    representations of data. Specifically, it represents both review groups
    and reviews as separate latent code types.
    """

    def __init__(self, arch_instance,  vocab_size, ext_vocab_size, emb_dim, enc_hidden_dim,
                 c_dim, z_dim, att_hidden_dim, dataset, use_se, kernel_features,
                 num_channels_enc, num_nf,num_preprocess_blocks, num_preprocess_cells, num_latent_scales, num_groups_per_scale, num_latent_per_group,
                 ada_groups, min_groups_per_scale, num_cell_per_cond_enc, num_channels_dec,
                 num_postprocess_blocks, num_postprocess_cells, num_cell_per_cond_dec, res_dist, num_mixture_dec, sig_depth,
                 states_sc_hidden=150, cgate_hidden_dim=None,  word_padding_idx=0, dec_layers =1, dec_hidden_size= 200):
        """
        :param vocab_size: the number of words by the generator.
        :param ext_vocab_size: the extended number of words that is accessible
            by the copy mechanism.
        :param emb_dim: the number of dimensions in word embeddings.
        :param enc_hidden_dim: GRU encoder's hidden dimension.
        :param c_dim: dimension of the group representation.
        :param z_dim: dimension of the review representation.
        :param att_hidden_dim: hidden dimension of hidden dimension of
            feed-forward networks.
        :param states_sc_hidden: hidden dimension of the score function used
            for production of the group representation c.
        :param cgate_hidden_dim: copy gate hidden dimension.
        """
        assert vocab_size <= ext_vocab_size

        super(CopyCat, self).__init__()
        # self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.bart_model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # self.bart_model.model.encoder.layers[3] = CustomBartLayer(self.get_z, self.bart_model.model.encoder.layers[3])
        # self.bart_lm.model.decoder.layers[3] = CustomBartLayer(self.get_z, self.bart_lm.model.decoder.layers[3])
        self.AvgPool = nn.AdaptiveAvgPool1d(1)

        self.emb_dim = emb_dim
        # self._embds = Embedding(ext_vocab_size, emb_dim)
        self._embds = self.bart_model.model.decoder.embed_tokens
        self.dataset = dataset
        self.arch_instance = arch_instance
        self.enc_hidden_dim = enc_hidden_dim

        self.use_se = use_se

        self.sig_depth = sig_depth
        # self.sig = Signature(self.sig_depth)


        self.res_dist = res_dist
        self.with_nf = num_nf > 0

        # encoder parameter
        self.kernel_features = kernel_features
        self.num_channels_enc = num_channels_enc
        self.num_channels_dec = num_channels_dec
        self.num_preprocess_blocks = num_preprocess_blocks  # block is defined as series of Normal followed by Down
        self.num_preprocess_cells = num_preprocess_cells  # number of cells per block
        self.num_cell_per_cond_enc = num_cell_per_cond_enc  # number of cell for each conditional in encoder

        # decoder parameters
        # self.num_channels_dec = args.num_channels_dec
        self.num_postprocess_blocks = num_postprocess_blocks
        self.num_postprocess_cells = num_postprocess_cells
        self.num_cell_per_cond_dec = num_cell_per_cond_dec  # number of cell for each conditional in decoder
        self.num_mix_output = num_mixture_dec

        self.num_latent_scales = num_latent_scales  # number of spatial scales that latent layers will reside
        self.num_groups_per_scale = num_groups_per_scale  # number of groups of latent vars. per scale
        self.num_latent_per_group = num_latent_per_group  # number of latent vars. per group
        self.groups_per_scale = groups_per_scale(self.num_latent_scales, self.num_groups_per_scale, ada_groups,
                                                 minimum_groups= min_groups_per_scale)
        self.vanilla_vae = self.num_latent_scales == 1 and self.num_groups_per_scale == 1

        # general cell parameters
        # self.input_size = get_input_size(self.dataset)
        # 100 60, 16 36
        self.input_size = 104

        # self.input_dim_lstm = signatory.logsignature_channels(self.num_channels_enc, self.sig_depth)

        # used for generative purpose
        c_scaling = CHANNEL_MULT ** (self.num_preprocess_blocks + self.num_latent_scales -1)
        self.spatial_scaling = 2 ** (self.num_preprocess_blocks + self.num_latent_scales -1)
        print(self.spatial_scaling, 'spatial_scaling*******************')
        prior_ftr0_size = (int(c_scaling * self.num_channels_dec), (self.input_size // self.spatial_scaling) // 2)
        # prior_ftr0_size = (int(c_scaling * self.num_channels_dec), (self.input_size // self.spatial_scaling) // 4)
        self.prior_ftro = Parameter(T.rand(size=prior_ftr0_size), requires_grad=True)

        self.AdaptiveMaxPool1d = nn.AdaptiveAvgPool1d(self.input_size // self.spatial_scaling)
        self.AvgPool = nn.AdaptiveAvgPool1d(1)
        if self.num_preprocess_cells % 2 == 0:
            self.AdaptiveMaxPool1d_stem = nn.AdaptiveMaxPool1d(int(4 * self.num_preprocess_cells))
        else:
            self.AdaptiveMaxPool1d_stem = nn.AdaptiveMaxPool1d(int(4 * (self.num_preprocess_cells+1)))

        self.num_flows = num_nf

        self.stem = self.init_stem()
        self.pre_process, mult = self.init_pre_process(mult=1)

        if self.vanilla_vae:
            self.enc_tower = []
        else:
            self.enc_tower, mult = self.init_encoder_tower(mult)

        self.enc0 = self.init_encoder0(mult)
        self.enc_sampler, self.dec_sampler, self.nf_cells, self.enc_kv, self.dec_kv, self.query = \
            self.init_normal_sampler(mult)

        if self.vanilla_vae:
            self.dec_tower = []
            self.stem_decoder = Conv1D(self.num_latent_per_group, mult * self.num_channels_enc, (1,1), bias=True)
        else:
            self.dec_tower, mult = self.init_decoder_tower(mult)

        self.post_process, mult = self.init_post_process(mult)

        self.conditional = self.init_conditional(mult)

        # self._encoder = Encoder(768, 1, n_head=8, d_k=96, d_v=96,
        #         d_model=dec_hidden_size, d_inner=ff_size, n_position=250, dropout=0.1,
        #         scale_emb=False)

        self._encoder = GruEncoder(input_dim=emb_dim, hidden_dim=enc_hidden_dim)
        self.enc_summary = GruEncoder(input_dim=768, hidden_dim=768)

        self.all_log_norm = []
        self.all_conv_layers = []
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            # if isinstance(layer, Conv2D) and '_ops' in n:   # only chose those in cell
            if isinstance(layer, Conv1D) or isinstance(layer, ARConv1d):
                self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)
            if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.SyncBatchNorm) or \
                    isinstance(layer, SyncBatchNormSwish):
                self.all_bn_layers.append(layer)
        print('len log norm:', len(self.all_log_norm))
        print('len bn:', len(self.all_bn_layers))
        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

        # POINTER-GENERATOR NETWORK #

        #   generation network - computes generation distribution over words
        # dec_inp_dim = z_dim + enc_hidden_dim
        # dec_inp_dim = 768 + enc_hidden_dim
        dec_inp_dim = 768 + 768
        # dec_inp_dim = z_dim
        gen_network = Sequential()
        gen_network.add_module("lin_proj", Linear(dec_inp_dim, emb_dim))
        gen_network.add_module("out_embds", OutEmbds(self._embds, vocab_size))
        gen_network.add_module("softmax", Softmax(dim=-1))

        #   copy gate - computes probability of copying a word
        c_ffnn = Ffnn(
                      # z_dim + enc_hidden_dim + emb_dim,
                      # 768 + enc_hidden_dim + emb_dim,
                      768 + 768 + emb_dim,
                      hidden_dim=cgate_hidden_dim,
                      non_linearity=Tanh(), output_dim=1)
        # c_ffnn = Ffnn(z_dim + emb_dim,
        #               hidden_dim=cgate_hidden_dim,
        #               non_linearity=Tanh(), output_dim=1)
        copy_gate = Sequential()
        copy_gate.add_module("ffnn", c_ffnn)
        copy_gate.add_module("sigmoid", Sigmoid())

        pgn = PointerGenNetwork(gen=gen_network, copy_gate=copy_gate,
                                ext_vocab_size=ext_vocab_size)

        # COMPLETE DECODER (GRU + ATTENTION + COPY-GEN) #
        #
        #   attention for decoder over encoder's hidden states
        dec_att = Attention(
                            # query_dim=z_dim,
                            query_dim=768,
                            hidden_dim=att_hidden_dim,
                            # value_dim=enc_hidden_dim,
                            value_dim=768,
                            non_linearity=Tanh())
        self._keys_creator = dec_att.create_keys

        self.linear_change = Linear(z_dim, 768)
        self.linear_cat = Linear(int(z_dim * self.input_size / 4), z_dim)
        self._decoder = GruPointerDecoder(
                                          # input_dim=emb_dim + z_dim,
                                          # hidden_dim=z_dim,
                                          input_dim=emb_dim + 768,
                                          hidden_dim=768,
                                          # contxt_dim=enc_hidden_dim,
                                          contxt_dim=768,
                                          att_module=dec_att,
                                          pointer_gen_module=pgn,
                                          # transformer_decoder=self.transformer_decoder,
                                          transformer_decoder=self.bart_model.model.decoder,
                                          word_padding_idx=0,
                                          cat_contx_to_inp=True,
                                          pass_extra_feat_to_pg=False)



    def init_stem(self):

        Cout = self.num_channels_enc
        Cin = self.enc_hidden_dim
        stem = Conv1D(Cin, Cout, 3, padding=1, bias=True)
        return stem


    # this part uses invertedResidual
    def init_pre_process(self, mult):
        pre_process = ModuleList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells -1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.num_channels_enc * mult)
                    num_co = int(CHANNEL_MULT * num_ci)
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch, use_se=self.use_se)
                    mult = CHANNEL_MULT * mult
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = self.num_channels_enc * mult
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch, use_se=self.use_se)
                pre_process.append(cell)
        return pre_process, mult

    def init_encoder_tower(self, mult):
        enc_tower = ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[s]):
                for c in range(self.num_cell_per_cond_enc):
                    arch = self.arch_instance['normal_enc']
                    num_c = int(self.num_channels_enc * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch, use_se=self.use_se)
                    enc_tower.append(cell)

                # add encoder combiner 这里应该是从r出来的z和h相加 正好和group个数对应 因为有个reverse
                if not (s == self.num_latent_scales - 1 and g == self.groups_per_scale[s] - 1):
                    num_ce = int(self.num_channels_enc * mult)
                    num_cd = int(self.num_channels_dec * mult)
                    cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    enc_tower.append(cell)

            # down cells after finishing a scale
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['down_enc']
                num_ci = int(self.num_channels_enc * mult)
                num_co = int(CHANNEL_MULT * num_ci)
                cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch, use_se=self.use_se)
                enc_tower.append(cell)
                mult = CHANNEL_MULT * mult
        return enc_tower, mult

    def init_encoder0(self, mult):
        num_c = int(self.num_channels_enc * mult)
        cell = Sequential(
            ELU(),
            Conv1D(num_c, num_c, kernel_size=1, bias=True),
            ELU())
        return cell
    def init_normal_sampler(self, mult):
        enc_sampler, dec_sampler, nf_cells = ModuleList(), ModuleList(), ModuleList()
        enc_kv, dec_kv, query = ModuleList(), ModuleList(), ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                # 刚开始编码的时候是从上往下，但是你看看现在这个从中采样是从上往下来，和图片的思想一样
                num_c = int(self.num_channels_enc * mult)
                # cell = Conv1D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
                cell = Conv1D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
                enc_sampler.append(cell)
            # build NF (normalizing flow) autoregressive flow, so it has lots of latent
                for n in range(self.num_flows):
                    arch = self.arch_instance['ar_nn']
                    num_c1 = int(self.num_channels_enc * mult)
                    num_c2 = 8 * self.num_latent_per_group
                    nf_cells.append(PairedCellAR(self.num_latent_per_group, num_c1, num_c2, arch))
                if not (s == 0 and g == 0): # for the first group, we use the a fixed standed normal
                    num_c = int(self.num_channels_dec * mult)
                    cell = Sequential(
                        ELU(),
                        Conv1D(num_c, 2 * self.num_latent_per_group, kernel_size=1, padding=0, bias=True))
                    dec_sampler.append(cell)

            mult = mult /CHANNEL_MULT
        return enc_sampler, dec_sampler, nf_cells, enc_kv, dec_kv, query

    def init_decoder_tower(self, mult):
        # create decoder tower
        dec_tower = ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                num_c = int(self.num_channels_dec * mult)
                if not (s == 0 and g == 0):
                    for c in range(self.num_cell_per_cond_enc):
                        arch = self.arch_instance['normal_dec']
                        cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch, use_se=self.use_se)
                        dec_tower.append(cell)

                cell = DecCombinerCell(num_c, self.num_latent_per_group, num_c, cell_type='combiner_dec')
                dec_tower.append(cell)

            # down cells after finishing a scale
            if s < self.num_latent_scales -1:
                print("s 执行没？？？？？？？？？？？？？？？？？")
                arch = self.arch_instance['up_dec']
                num_ci = int(self.num_channels_dec * mult)
                num_co = int(num_ci / CHANNEL_MULT)
                cell = Cell(num_ci, num_co, cell_type='up_dec', arch=arch, use_se=self.use_se)
                dec_tower.append(cell)
                mult = mult / CHANNEL_MULT
        return dec_tower, mult

    def init_post_process(self, mult):
        post_process = ModuleList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.num_channels_dec * mult)
                    num_co = int(num_ci / CHANNEL_MULT)
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch, use_se=self.use_se)
                    mult = mult / CHANNEL_MULT
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.num_channels_dec * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch, use_se=self.use_se)

                post_process.append(cell)

        return post_process, mult

    def init_conditional(self, mult):
        C_in = int(self.num_channels_dec / 2)
        C_out = int(self.num_channels_dec)
        return Sequential(ELU(),
                          Conv1D(C_in, C_out, 3, padding=1, bias=True))



    def forward(self, rev, rev_len, rev_mask,
                # prompt, prompt_len,
                group_rev_indxs, group_rev_indxs_mask,
                rev_to_group_indx, other_rev_indxs, other_rev_indxs_mask,
                other_rev_comp_states, other_rev_comp_states_mask,
                c_lambd=0., z_lambd=0.):
        """
        :param rev: review word ids.
            [batch_size, rev_seq_len]
        :param rev_len: review lengths.
            [batch_size]
        :param rev_mask: float mask where 0. is set to padded words.
            [batch_size, rev_seq_len]
        :param rev_to_group_indx: mapping from reviews to their corresponding
            groups.
            [batch_size]
        :param group_rev_indxs: indxs of reviews that belong to same groups.
            [group_count, max_rev_count]
        :param group_rev_indxs_mask: float mask where 0. is set to padded
            review indxs.
            [group_count, max_rev_count]
        :param other_rev_indxs: indxs of leave-one-out reviews.
            [batch_size, max_rev_count]
        :param other_rev_indxs_mask: float mask for leave-one-out reviews.
        :param other_rev_comp_states: indxs of (hidden) states of leave-one-out
            reviews. Used as an optimization to avoid attending over padded
            positions.
            [batch_size, cat_rev_len]
        :param other_rev_comp_states_mask: masking of states for leave-one-out
            reviews.
            [batch_size, cat_rev_len]
        :param c_lambd: annealing constant for c representations.
        :param z_lambd: annealing constant for z representations.

        :return loss: scalar loss corresponding to the mean ELBO over batches.
        :return metrs: additional statistics that are used for analytics and
            debugging.
        """
        bs = rev.size(0)
        device = rev.device
        group_count = group_rev_indxs.size(0)
        loss = 0.
        metrs = OrderedDict()
        rev_word_embds = self._embds(rev)

        rev_encs, rev_hiddens = self.encode(rev_word_embds, rev_len)
        # rev_hiddens = self.encode(rev_word_embds, rev_mask)

        rev_code = T.transpose(rev_hiddens, 2, 1)
        z_hiddens, all_q, all_p, all_log_q, all_log_p, collect_z = self.get_z(rev_code)



        final_rev_len = T.ones(z_hiddens.size()[0])
        final_rev_len = final_rev_len * z_hiddens.size()[1]
        z, _ = self.enc_summary(z_hiddens, final_rev_len)
        att_keys = self.create_att_keys(rev_hiddens)


        rev_att_keys, \
        rev_att_vals, \
        rev_att_mask = group_att_over_input_rev(inp_att_keys=att_keys,
                                            # inp_att_vals=first_rev_hiddens,
                                            inp_att_vals=rev_hiddens,
                                            inp_att_mask=rev_mask,
                                            att_indxs=other_rev_indxs,
                                            att_indxs_mask=other_rev_indxs_mask)


        rev_att_word_ids = rev[other_rev_indxs].view(bs, -1)



        # optimizing the attention targets by making more compact tensors
        # with less padded entries
        sel = T.arange(bs, device=device).unsqueeze(-1)
        rev_att_keys = rev_att_keys[sel, other_rev_comp_states]
        rev_att_vals = rev_att_vals[sel, other_rev_comp_states]
        rev_att_mask = other_rev_comp_states_mask
        rev_att_word_ids = rev_att_word_ids[sel, other_rev_comp_states]

        # creating an extra feature that is passe
        extra_feat = z.unsqueeze(1).repeat(1, rev_word_embds.size(1), 1)

        log_probs, rev_att_wts, \
        hidden, cont, \
        copy_probs = self._decode(embds=rev_word_embds, mask=rev_mask,
                                  extra_feat=extra_feat,
                                  hidden=z, att_keys=rev_att_keys,
                                  att_values=rev_att_vals,
                                  att_mask=rev_att_mask,
                                  att_word_ids=rev_att_word_ids)


        rec_term = comp_seq_log_prob(log_probs[:, :-1], seqs=rev[:, 1:],
                                     seqs_mask=rev_mask[:, 1:])
        avg_rec_term = rec_term.mean(dim=0)

        loss += -avg_rec_term
        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            if self.with_nf:
                kl_per_var = log_q_conv - log_p_conv
            else:
                kl_per_var = q.kl(p)


            kl_diag.append(T.mean((T.sum(kl_per_var, dim=2)),dim=0))
            kl_all.append(T.sum(kl_per_var, dim=[1, 2]))
            log_q += T.sum(log_q_conv, dim=[1, 2])
            log_p += torch.sum(log_p_conv, dim=[1, 2])

        metrs['avg_rec'] = avg_rec_term.item()

        # print(metrs['avg_rec'],'cehck -==-=-=-=-=-')

        return loss, kl_all, kl_diag, metrs

    def get_final_output(self, layer_num=3, rev=None, input_emb=None, summary=False):

        if summary:

            enc_out = self.bart_model.model.encoder(inputs_embeds=input_emb)
            latent_code = enc_out.last_hidden_state
            return latent_code
            # return beam_input
        else:
            out_put= self.bart_model.model(input_ids=rev, output_hidden_states=True)
            p_q = self.bart_model.model.encoder.layers[layer_num].p_q
            down_normal = self.bart_model.model.encoder.layers[layer_num].down_normal
            return out_put, p_q, down_normal



    def get_z(self, s):

        s = self.AdaptiveMaxPool1d(s)

        s = self.stem(s)

        # perform pre-processing
        for cell in self.pre_process:
            s = cell(s)


        # s = self.AdaptiveMaxPool1d(s)

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)

        # reverse combiner cells and their input for decoder
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec = 0
        ftr = self.enc0(s)  # this reduces the channel dimension
        param0 = self.enc_sampler[idx_dec](ftr)
        mu_q, log_sig_q = T.chunk(param0, 2, dim=1)  # 按照channel分，就是维度，分成两份
        dist = Normal(mu_q, log_sig_q)  # for the first approx. posterior
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)

        # apply normalizing flows
        nf_offset = 0
        for n in range(self.num_flows):
            z, log_det = self.nf_cells[n](z, ftr)
            log_q_conv -= log_det
        nf_offset += self.num_flows
        all_q = [dist]
        all_log_q = [log_q_conv]

        # To make sure we do not pass any deterministic features from x to decoder.
        s = 0

        dist = Normal(mu=T.zeros_like(z), log_sigma=T.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]
        collect_z = list()

        idx_dec = 0
        s = self.prior_ftro.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1)
        similarity_z_rev = list()
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = T.chunk(param, 2, dim=1)

                    # form encoder
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    # print(ftr.size(), 'ftr size -=-=-=-=- the second combination between s and z')
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = T.chunk(param, 2, dim=1)
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.res_dist else Normal(mu_q, log_sig_q)
                    z, _ = dist.sample()
                    log_q_conv = dist.log_p(z)
                    collect_z.append(T.transpose(z, 2, 1))
                    # apply NF
                    for n in range(self.num_flows):
                        z, log_det = self.nf_cells[nf_offset + n](z, ftr)
                        log_q_conv -= log_det
                    nf_offset += self.num_flows
                    all_log_q.append(log_q_conv)
                    all_q.append(dist)


                    # evaluate log_p(z)
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                # combiner_dec
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)


        mu_q, log_sig_q = T.chunk(s, 2, dim=1)
        final_dist = Normal(mu_q, log_sig_q)
        z, _ = final_dist.sample()
        log_q_conv = final_dist.log_p(z)
        z = self.conditional(log_q_conv)
        z = T.transpose(z, 2, 1)
        return z, all_q, all_p, all_log_q, all_log_p, collect_z



    def get_p_z_mu_sigma(self, c):
        """Computes z prior's parameters (mu, sigma)."""
        mu_p, sigma_p = self._z_prior_network(c)
        return mu_p, sigma_p

    def get_q_z_mu_sigma(self, encs, c):
        """
        Runs the inference network and computes mu and sigmas for review latent
        codes (z).
        """
        inp = T.cat((encs, c), dim=-1)
        mu_q, sigma_q = self._z_inf_network(inp)
        return mu_q, sigma_q

    def get_q_c_mu_sigma(self, states, states_mask, group_indxs,
                         group_indxs_mask):
        """Computes c approximate posterior's parameters (mu, sigma).

        :param states: [batch_size, seq_len1, dim]
                        representations of review steps, e.g. hidden + embd.
        :param states_mask: [batch_size, seq_len1]
        :param group_indxs: [batch_size2, seq_len2]
                      indxs of reviews belonging to the same product
        :param group_indxs_mask: [batch_size2, seq_len2]
        """
        grouped_states, \
        grouped_mask = group_att_over_input(inp_att_vals=states,
                                            inp_att_mask=states_mask,
                                            att_indxs=group_indxs,
                                            att_indxs_mask=group_indxs_mask)
        ws_state, \
        score_weights = self._compute_ws_state(states=grouped_states,
                                               states_mask=grouped_mask)
        mu, sigma = self._c_inf_network(ws_state)

        return mu, sigma, score_weights

    def encode(self, embds, mask_or_lens):
        return self._encoder(embds, mask_or_lens)
        # return self._encoder(embds, mask_or_lens)

    def decode_beam(self, seqs, hidden, att_word_ids, init_z=None, **kwargs):
        """Function to be used in the beam search process.

        :param seqs: [batch_size, 1]
        :param hidden: [batch_size, hidden_dim]
        :param att_word_ids: [batch_size, cat_rev_len]
        :param init_z: [batch_size, z_dim]
        """

        embds = self._embds(seqs)
        mask = T.ones_like(seqs, dtype=T.float32)

        if init_z is None:
            init_z = hidden

        word_log_probs, att_wts, \
        hidden, cont, ptr_probs = self._decode(embds=embds, mask=mask,
                                               extra_feat=init_z.unsqueeze(1),
                                               hidden=hidden,
                                               att_word_ids=att_word_ids,
                                               **kwargs)
        out = DecState(word_scores=word_log_probs,
                       rec_vals={"hidden": hidden, "cont": cont,
                                 "init_z": init_z},
                       coll_vals={'copy_probs': ptr_probs.squeeze(-1),
                                  'att_wts': att_wts.squeeze(1),
                                  "att_word_ids": att_word_ids})
        return out

    def _decode(self, embds, mask, hidden, cont=None, eps=EPS, **dec_kwargs):
        """Teacher forcing decoding of sequences.

        :param embds: [batch_size, seq_len, dim]
        :param mask: [batch_size, seq_len]
        :param att_word_ids: [batch_size, inp_seq_len]
        :param copy_prob: a fixed probability of copying words.
        :param hidden: [batch_size, hidden_dim]
        """
        word_probs, copy_probs, \
        att_weights, prev_hidden, \
        prev_cont = self._decoder(embds, mask, init_hidden=hidden,
                                  init_cont=cont, **dec_kwargs)

        word_log_probs = T.log(word_probs + eps)

        return word_log_probs, att_weights, prev_hidden, prev_cont, copy_probs

    def _compute_ws_state(self, states, states_mask):
        """Computes weighted state by scoring each state."""
        state_scores = self._c_states_scoring(states)
        grouped_state_weights = masked_softmax(state_scores, states_mask, dim=1)
        group_context = (states * grouped_state_weights).sum(dim=1)
        return group_context, grouped_state_weights

    def create_att_keys(self, hiddens):
        """
        Creates the attention keys that are used in the decoder.
        Performs projection that is required by the attention mechanism, allows
        for a speed-up by not performing this operation every time attention is
         called (at each decoding step).

        :param hiddens: [batch_size, seq_len, hidden_dim]
        """
        att_keys = self._keys_creator(hiddens)
        return att_keys

    def get_contxt_states(self, hiddens, embds):
        return T.cat((hiddens, embds), dim=-1)

    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))

        return loss

    def get_attention(self, z, prom_hiddens):

        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        z_len = z.size(1)
        z_batch = z.size(0)
        prom_len = prom_hiddens.size(1)

        # strategy 2
        for i in range(z_len):
            if i == 0:
                sub_z = z[:, i, :].unsqueeze(1).repeat(1, prom_len, 1)
                # [batch_size, prm_len, 1]
                sim_attn = cos(sub_z, prom_hiddens)
                sim_attn = T.sum(sim_attn, dim=-1).unsqueeze(-1).unsqueeze(-1)
            else:
                sub_z = z[:, i, :].unsqueeze(1).repeat(1, prom_len, 1)
                # [batch_size, prm_len, 1]
                sub_attn = cos(sub_z, prom_hiddens)
                sub_attn = T.sum(sub_attn, dim=-1).unsqueeze(-1).unsqueeze(-1)
                sim_attn = T.cat((sim_attn, sub_attn), dim=1)

        sim_attn = F.softmax(sim_attn, dim=1)
        z_final = T.mul(z, sim_attn)
        # 这个z_final 的size is[batch_size, dimention] 不用再进行encode操作
        z_final = T.sum(z_final, dim=1).squeeze(1)
        return z_final

