
# from ..utils.constants import VOCAB_DEFAULT_SYMBOLS
from thvae.utils.constants import VOCAB_DEFAULT_SYMBOLS
from thvae.data_pipelines.assemblers import assemble_vocab_pipeline
from thvae.utils.fields import InpDataF
import argparse
from mltoolkit.mlutils.helpers.logging import init_logger
from mltoolkit.mldp.utils.tools import Vocabulary


logger = init_logger("")


def create_prompt(vocab_fp, data_path, sep='\t'):
    """Creates a word vocabulary using a simple pipeline."""
    vocab_pipeline = assemble_vocab_pipeline(text_fname=InpDataF.PROMPT,
                                             sep=sep)
    words_vocab = Vocabulary(vocab_pipeline, name_prefix="prompt")


    # adding special symbols before creating vocab, so they would appear on top
    for st in VOCAB_DEFAULT_SYMBOLS:
        if st not in words_vocab:
            words_vocab.add_special_symbol(st)

    words_vocab.create(data_source={'data_path': data_path},
                       data_fnames=InpDataF.PROMPT)
    words_vocab.write(vocab_fp, sep=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_fp", type=str,
                        help='Output file location of the create vocabulary.')
    parser.add_argument('--data_path', type=str, nargs='+',
                        help='Directory or a file with data.')
    parser.add_argument('--sep', type=str, default='\t')
    create_prompt(**vars(parser.parse_args()))
