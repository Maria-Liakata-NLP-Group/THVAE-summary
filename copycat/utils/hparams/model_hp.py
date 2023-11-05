from mltoolkit.mlmo.utils.tools import BaseHP


class ModelHP(BaseHP):
    """
    Contains hyper-parameters of the actual model.
    Please see `CopyCat` for more details on hyper-parameters.
    """

    def __init__(self):
        super(ModelHP, self).__init__()
        # self.vocab_size = 21830
        self.vocab_size = 50265
        self.ext_vocab_size = 50265
        # self.emb_dim = 768
        # self.emb_dim = 286
        # self.enc_hidden_dim = 384
        self.enc_hidden_dim = 768
        # self.emb_dim = 128
        self.emb_dim = 768
        # 这个要和signature的输出的维度保持一致，因为还有最终的summary 生成
        # self.enc_hidden_dim = 384
        self.c_dim = 600
        self.z_dim = 768
        # self.z_dim = 385
        # self.z_dim = 384
        self.states_sc_hidden = 512
        # self.att_hidden_dim = 256
        self.att_hidden_dim = 512
        self.cgate_hidden_dim = 128
        # self.num_channels_enc = 385
        # self.num_channels_enc = 384
        self.num_channels_enc = 768
        # self.num_channels_enc = 10
        # self.num_channels_enc = 768

        # transformer parameter
        self.dec_layers = 1
        self.dec_hidden_size = 768
        # self.heads = 8
        # self.ff_size = 3072

        self.dataset = ''
        # The number of normalizing flow cells per groups. Set this to zero to disable flows.
        self.num_nf = 0
        # self.kernel_features = [68, 125, 125, 200, 250]
        self.kernel_features = [25, 50, 75, 100, 134]
        self.num_preprocess_blocks = 1
        self.num_preprocess_cells = 3
        self.num_latent_scales = 1  # the number of latent scales
        self.num_groups_per_scale = 5  # number of groups of latent variables per scale
        self.num_latent_per_group = 768  # number of channels in latent variables per group
        self.ada_groups = False # Settings this to true will set different number of groups per scale.
        self.min_groups_per_scale = 1  # the minimum number of groups per scale.
        self.num_cell_per_cond_enc = 1 # number of cell for each conditional in encoder

        # decoder parameters
        # self.num_channels_dec = 10
        # self.num_channels_dec = 385
        # self.num_channels_dec = 384
        self.num_channels_dec = 768
        # self.num_channels_dec = 768  # number of channels in decoder
        self.num_postprocess_blocks = 1
        self.num_postprocess_cells = 3
        self.num_cell_per_cond_dec = 1
        self.num_mixture_dec = 8  # number of mixture components in decoder. set to 1 for Normal decoder.

        # This flag enables squeeze and excitation.
        self.res_dist = False
        self.use_se = True

        # signature
        self.sig_depth = 3

