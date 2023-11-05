from mltoolkit.mlmo.utils.tools import BaseHP
from mltoolkit.mlutils.helpers.paths_and_files import comb_paths
from mltoolkit.mlutils.tools import ExperimentsPathController
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesTruecaser
from functools import partial
import nltk
from mltoolkit.mlmo.utils.helpers.analytics import ngram_seq_analysis


class RunHP(BaseHP):
    """Contains configuration parameters for running the model."""

    def __init__(self):
        super(RunHP, self).__init__()

        self.experiments_descr = 'My first experiment with the model.'

        #   GENERAL  #
        self.seed = 42
        self.cuda_device_id = 0
        self.device = 'cpu'  # 'cuda' or 'cpu'
        self.training_logging_step = 50  # that often to print internal metrics
        self.epochs = 0  # if set to 0 wi ll only perform evaluation
        self.learning_rate = 0.0005
        # self.learning_rate = 1e-2
        self.grads_clip = 0.25

        # GENERAL DATA RELATED #
        self.dataset = 'amazon'
        # self.train_max_groups_per_batch = 3
        # self.val_max_groups_per_batch = 7
        # self.eval_max_groups_per_batch = 10
        # self.max_rev_per_group = 8
        self.train_max_groups_per_batch = 2
        self.val_max_groups_per_batch = 2
        self.eval_max_groups_per_batch = 3
        self.max_rev_per_group = 4

        #   DATA SOURCES  #
        # `early_term` limits the numberpython copycat/scripts/run_workflow.py of chunks per epoch
        # set it to ``None`` to validate on the full set
        self.train_early_term = None
        self.val_early_term = 500
        self.gener_early_term = 2

        #  GENERAL PATHS   #
        self.root_path = 'copycat'
        self.experiments_folder = 'first_run'
        self.output_dir = f'{self.root_path}/runs/{self.dataset}/{self.experiments_folder}'
        self.checkpoint_full_fn = 'checkpoint.tar'
        epc = ExperimentsPathController()
        self.output_path = epc(self.output_dir)
        # self.checkpoint_path = f'{self.root_path}/artifacts/{self.dataset}/checkpoint.tar'
        self.checkpoint_path = ''
        # self.tcaser_model_path = f'{self.root_path}/artifacts/{self.dataset}/data/tcaser.model'
        # self.tcaser_model_path = 'copycat/artifacts/amazon/data/tcaser.model'
        self.tcaser_model_path = 'data/dataset_name/tcaser.model'

        #   DATA PATHS  #
        # self.base_data_path = f'data/{self.dataset}/'
        # self.train_fp = comb_paths(self.base_data_path, "split/train/")
        # self.train_fp = "/home/qinghua/jsong/wass-transformer/data/amazon/4.part/train"
        # self.val_fp = "/home/qinghua/jsong/wass-transformer/data/amazon/4.part/val"
        # self.train_fp = "data/amazon/split-mum/train"
        # self.val_fp = "data/amazon/split-mum/val"
        # self.train_fp = "data/amazon/reddit/train"
        self.train_fp = "data/amazon/reddit_prompt/train"
        # self.train_fp = "data/amazon/tweet/train"
        # self.train_fp = "data/amazon/split/train"
        # self.train_fp_1 = "data/amazon/train"
        # self.val_fp = "data/amazon/val"
        self.val_fp = "data/amazon/reddit_prompt/val"
        # self.val_fp = "data/amazon/tweet/val"
        # self.val_fp = "data/amazon/split/val"
        # self.words_vocab_fp = f'{self.root_path}/artifacts/{self.dataset}/data/words.txt'
        # self.words_vocab_fp = 'copycat/artifacts/amazon/data/words.txt'
        self.words_vocab_fp = 'data/dataset_name/vocabs/vocab.txt'
        # self.eval_dev_fp = "/home/qinghua/jsong/wass_DSS/data/amazon/gold3/val.csv"
        # self.eval_test_fp = "/home/qinghua/jsong/wass_DSS/data/amazon/gold3/test.csv"
        self.eval_dev_fp = "data/amazon/gold_8/val.csv"
        self.eval_test_fp = "data/amazon/gold_8/test.csv"
        # self.eval_dev_fp = "data/amazon/gold/key_phrase.csv"
        # self.eval_test_fp = "data/amazon/gold/key_phrase.csv"
        # self.eval_dev_fp = "data/amazon/gold/val_t.csv"
        # self.eval_test_fp = "data/amazon/gold/test_t.csv"
        self.gen_fp1 = 'Timeline_dataset/timeline_test1'
        self.gen_fp2 = 'Timeline_dataset/timeline_test2'
        self.gen_fp = 'Timeline_dataset/timeline_test'
        self.gen_fp_add = 'Timeline_dataset/timeline_test_add'
        self.gen_fp3 = 'Timeline_dataset/timeline_test3'
        self.gen_fp4 = 'Timeline_dataset/timeline_test4'
        self.gen_fp5 = 'Timeline_dataset/timeline_test5'
        self.gen_fp6 = 'Timeline_dataset/timeline_test6'
        self.gen_fp7 = 'Timeline_dataset/timeline_test7'
        self.gen_add4 = 'Timeline_dataset/timeline_add_4'

        #   ANNEALING   #
        self.c_m = 8.
        self.c_r = 0.8
        self.c_kl_ann_max_val = 1.
        self.c_kl_ann_batches = self.epochs * self.train_early_term if self.train_early_term else self.epochs * 10000
        self.z_m = 8.
        self.z_c = 0.8
        self.z_kl_ann_max_val = 1.
        self.z_kl_ann_batches = self.epochs * self.train_early_term if self.train_early_term else self.epochs * 10000

        self.kl_anneal_portion = 0.3 # The portions epochs that KL is annealed
        self.kl_const_portion = 0.0001 # The portions epochs that KL is constant at kl_const_coeff
        self.kl_const_coeff = 0.0001 # The constant value used for min KL coeff
        # This flag enables annealing the lambda coefficient from ''--weight_decay_norm_init to --weight_decay_norm.
        self.weight_decay_norm_anneal = True
        self.weight_decay_norm_init = 8. # the initial lambda parameter
        # The lambda parameter for spectral regularization.
        self.weight_decay_norm = 0.8


        #   DECODING/GENERATION  #
        self.beam_size = 7
        self.beam_len_norm = True
        self.beam_excl_words = []
        self.block_ngram_repeat = 3  # or None
        self.ngram_mirror_window = 3  # or None
        self.mirror_conjs = ["and", 'or', ',', 'but']  # or None
        self.block_consecutive = True
        self.min_gen_seq_len = 500

        #   POST-PROCESSING AND ANALYTICS #
        mt = MosesTokenizer()
        self.tok_func = partial(mt.tokenize, escape=False)
        self.sent_split_func = nltk.sent_tokenize
        dt = MosesDetokenizer()
        self.detok_func = partial(dt.detokenize, unescape=False)
        true_caser = MosesTruecaser(load_from=self.tcaser_model_path,
                                    is_asr=True)
        self.true_case_func = partial(true_caser.truecase, return_str=True,
                                      use_known=True)
        self.analytics_func = partial(ngram_seq_analysis,
                                      tokenizer=self.tok_func,
                                      sent_splitter=self.sent_split_func,
                                      n_grams_to_comp=(2, 3, 4))
