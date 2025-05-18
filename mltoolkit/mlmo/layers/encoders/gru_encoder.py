from torch.nn import Module, GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch as T

class GruEncoder(Module):
    """
    GRU encoder inputs embedded sequences and outputs their representations,
    which are concatenated hidden states. Optionally returns all hidden states.

    Works both if sequence lens or mask is provided, which allows propagation
    of gradient wrt to the (soft) mask.

    Notice that the relaxation copies the previous hidden layer instead of
    setting it to zeros.
    """

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(GruEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = GRU(input_dim, hidden_dim, bidirectional=False,
                       batch_first=True, **kwargs)

    def forward(self, x, lens_or_mask):
        """
        :param x: [batch_size, seq_len, input_dim]
                  embedded input sequences.
        :param lens_or_mask: [batch_size] or [batch_size, seq_len]
        :return  last hidden: [batch_size, hidden_dim]
                 all hiddens (opt): [batch_size, seq_len, hidden_dim]
        """
        batch_size, max_seq_len, _ = x.size()
        device = x.device

        if lens_or_mask.dim() == 1:
            # sequence lengths are passed
            packed_inp = pack_padded_sequence(
                x,
                lens_or_mask.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_out, last_hidden = self.gru(packed_inp)

            # ** wrap this in try/except to catch empty‐input errors **
            try:
                out, _ = pad_packed_sequence(packed_out, batch_first=True)
            except RuntimeError:
                print('GRU: empty input/ return zeros')
                # if packed_out had zero data, pad everything to zeros
                out = T.zeros(
                    (batch_size, max_seq_len, self.hidden_dim),
                    device=device,
                    dtype=x.dtype
                )
        else:
            # sequence mask is passed
            out = T.empty((batch_size, max_seq_len, self.hidden_dim),
                          device=device, dtype=x.dtype)
            prev_h = T.zeros((1, batch_size, self.hidden_dim), device=device)
            mask = lens_or_mask.unsqueeze(-1)
            for t in range(max_seq_len):
                _x = x[:, t].unsqueeze(1)
                _m = mask[:, t].unsqueeze(0)
                _, new_h = self.gru(_x, prev_h)
                # copy previous hidden if masked
                prev_h = _m * new_h + (1. - _m) * prev_h
                out[:, t] = prev_h.squeeze(0)
            last_hidden = prev_h

        last_hidden = last_hidden.squeeze(0)
        return last_hidden, out
