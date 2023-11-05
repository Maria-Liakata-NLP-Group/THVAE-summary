from mltoolkit.mlmo.interfaces import ITorchModel
from copycat.utils.fields import ModelF
from logging import getLogger
from mltoolkit.mlmo.utils.tools import DecState
import torch as T
import os
from copycat.utils.helpers.modelling import group_att_over_input, group_att_over_input_rev
import numpy as np
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer, StoppingCriteriaList, MaxTimeCriteria, MaxLengthCriteria
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class ICopyCat(ITorchModel):
    """Model interface. Contains a custom method for summary generation."""

    def __init__(self, beamer, min_gen_seq_len=None, **kwargs):
        """
        :param beamer: beam search object for sequence generation.
        :param min_gen_seq_len: minimum length of generated reviews and summaries.
        """
        super(ICopyCat, self).__init__(**kwargs)
        self.beamer = beamer
        self.min_sen_seq_len = min_gen_seq_len

    def predict(self, batch, **kwargs):
        """Predicts summaries and reviews. Used in development."""
        self.model.eval()
        group_id = batch[ModelF.GROUP_ID]
        revs = batch[ModelF.REV].to(self.device)
        rev_lens = batch[ModelF.REV_LEN].to(self.device)
        revs_mask = batch[ModelF.REV_MASK].to(self.device)
        prompt = batch[ModelF.PROMPT].to(self.device)
        prompt_lens = batch[ModelF.PROMPT_LEN].to(self.device)
        prompt_mask = batch[ModelF.PROMPT_MASK].to(self.device)
        summ_rev_indxs = batch[ModelF.GROUP_REV_INDXS].to(self.device)
        summ_rev_indxs_mask = batch[ModelF.GROUP_REV_INDXS_MASK].to(self.device)
        other_revs = batch[ModelF.OTHER_REV_INDXS].to(self.device)
        other_revs_mask = batch[ModelF.OTHER_REV_INDXS_MASK].to(self.device)
        rev_to_group_indx = batch[ModelF.REV_TO_GROUP_INDX].to(self.device)

        other_revs_mask = T.ones_like(other_revs_mask)

        bs = revs.size(0)
        max_rev_len = revs.size(1)
        summs_nr = summ_rev_indxs.size(0)

        with T.no_grad():

            # # bart decoder
            # z_output, _ = self.model.get_final_output(rev=revs)
            # z_decoder = z_output.last_hidden_state
            # z_latent = z_output.encoder_last_hidden_state
            # # DECODING OF SUMMARIES #
            # summary_latent = z_latent[summ_rev_indxs]
            # bs = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_z = summary_latent.view(bs, group_size * seq_len, -1)
            # summary = self.model.get_final_output(input_emb=summary_z, summary=True)
            #
            # _, rev_id = self.translate_sentence(z_latent)
            # _, summary_id = self.translate_sentence(summary)

            # rev_batch = revs.size(0)
            # diff_len = revs.size(1) - prompt.size(1)
            # diff_tensor = T.ones(size=(rev_batch, diff_len), dtype=T.int64).to(self.device)
            # prompt_c = T.cat((prompt, diff_tensor), dim=-1)
            # prompt_c = prompt_c.to(self.device)




            # normal decoder
            rev_embds = self.model._embds(revs)
            pro_embds = self.model._embds(prompt)
            # pro_c_embds = self.model._embds(prompt_c)



            # z_output, _, first_rev_hiddens = self.model.get_final_output(rev=revs)
            #
            # rev_hiddens = z_output.encoder_last_hidden_state
            # # first_rev_hiddens = z_output.encoder_hidden_states[4]
            #
            # att_keys = self.model.create_att_keys(first_rev_hiddens)
            # summ_att_word_ids = revs[summ_rev_indxs].view(summs_nr, -1)
            #
            # summary_latent = rev_hiddens[summ_rev_indxs]
            # bs_summary = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_z = summary_latent.view(bs_summary, group_size * seq_len, -1)
            # summary = self.model.get_final_output(input_emb=summary_z, summary=True)
            #
            # # final_rev_len = T.ones(summary.size()[0])
            # # final_rev_len = final_rev_len * summary.size()[1]
            # # summary, _ = self.model.enc_summary(summary, final_rev_len)
            # summary = T.transpose(summary, 2, 1)
            # summary = self.model.AvgPool(summary)
            # summary = summary.squeeze(-1)

            # out_put = self.model.bart_model.model.encoder(input_ids=revs)
            # rev_hiddens = out_put.last_hidden_state

            rev_encs, rev_hiddens = self.model.encode(rev_embds, rev_lens)
            # this encode is transformer encode
            # rev_hiddens = self.model.encode(rev_embds, revs_mask)

            pro_encs, pro_hiddens = self.model.enc_summary(pro_embds, prompt_lens)
            # _, pro_attn = self.model.encode(pro_c_embds, rev_lens)


            rev_code = T.transpose(rev_hiddens, 2, 1)
            z_hiddens, _, _, _, _, collect_z = self.model.get_z(rev_code)
            # z, _ = self.model.z_atten(z)
            # z_latent = z.unsqueeze(1)
            # summary_latent = z[summ_rev_indxs]
            # bs_summary = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_z = summary_latent.reshape(bs_summary, group_size * seq_len, -1)
            # summary_z = T.transpose(summary_z, 2, 1)
            # summary, _, _, _, _, collect_z = self.model.get_z(summary_z)

            # summary_latent = rev_encs[summ_rev_indxs]
            # summary_latent = pro_encs[summ_rev_indxs]
            # print(summary_latent.size(), 'size-=-=-=-=-=-=-=-')
            # summary_z = T.transpose(summary_latent, 2, 1)

            # summary_latent = pro_encs[summ_rev_indxs]
            # bs_summary = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_latent = summary_latent.reshape(bs_summary, group_size * seq_len, -1)

            # decode_output = self.model.bart_model.model.decoder(inputs_embeds=summary)
            # bart_hidden = decode_output.last_hidden_state
            # _, summary_id = self.translate_sentence_beamsearch(bart_hidden)

            # summary, _ = self.model.z_atten(summary)

            # summary = T.transpose(summary, 2, 1)
            # summary = self.model.AvgPool(summary)
            # summary = summary.squeeze(-1)

            ###################################################################### strategy 2:#############################

            r_len = rev_hiddens.size(1)
            pro_unsq = T.unsqueeze(pro_encs, dim=1)
            pro_repeat = pro_unsq.repeat(1, r_len, 1)

            sim = T.nn.CosineSimilarity(dim=2)
            sim_value = sim(rev_embds, pro_repeat)
            m = T.nn.Softmax(dim=1)
            sim_softmax = m(sim_value)
            sim_softmax = T.unsqueeze(sim_softmax, dim=-1)
            rev_val = T.mul(rev_embds, sim_softmax)
            val_encs, val_hiddens = self.model.encode(rev_val, rev_lens)

            val_group = val_encs[summ_rev_indxs]

            # rev_group = val_hiddens[summ_rev_indxs]
            #
            # bs_rev = rev_group.size()[0]
            # group_rev = rev_group.size()[1]
            # rev_len = rev_group.size()[2]
            # rev_rep = rev_group.reshape(bs_rev, group_rev * rev_len, -1)


            rev_trans = T.transpose(val_group, 2, 1)
            print(rev_trans.size(), 'cehck size -=-=-=-=-=-=-')
            # print(summary_z.size(), 'cehck z0-=-=-=-==-')
            rev_latent, _, _, _, _, _ = self.model.get_z(rev_trans)


            final_rev_len = T.ones(rev_latent.size()[0])
            final_rev_len = final_rev_len * rev_latent.size()[1]
            summary, _ = self.model.enc_summary(rev_latent, final_rev_len)

            ################################################### normal decoder from here ##############################################

            # rr = F.conv1d(T.transpose(rev_hiddens, 2, 1), weight=T.transpose(pro_hiddens, 2, 1))
            # print(rr.size(), 'chekc rr 0-=-=--=-=-=-=-')


            att_keys = self.model.create_att_keys(rev_hiddens)
            # sum_att_keys = self.model.create_att_keys(val_hiddens)
            summ_att_word_ids = prompt[summ_rev_indxs].view(summs_nr, -1)


            summ_att_keys, \
            summ_att_vals, \
            summ_att_mask = group_att_over_input(inp_att_keys=att_keys,
                                                 # inp_att_vals=first_rev_hiddens,
                                                 inp_att_vals=rev_hiddens,
                                                 inp_att_mask=revs_mask,
                                                 # inp_att_vals=pro_hiddens,
                                                 # inp_att_mask=prompt_mask,
                                                 att_indxs=summ_rev_indxs,
                                                 att_indxs_mask=summ_rev_indxs_mask)

            # DECODING OF SUMMARIES 
            if self.min_sen_seq_len is not None:
                # min_lens = [self.min_sen_seq_len] * batch_size
                min_lens = [self.min_sen_seq_len] * summs_nr
            else:
                min_lens = None

            # summary, _ = self.model.z_atten(summary)
            init_summ_dec_state = DecState(rec_vals={"hidden": summary})
            summ_word_ids, \
            summ_coll_vals = self.beamer(init_summ_dec_state,
                                         min_lens=min_lens,
                                         max_steps=max_rev_len,
                                         att_keys=summ_att_keys,
                                         att_values=summ_att_vals,
                                         att_mask=summ_att_mask,
                                         att_word_ids=summ_att_word_ids,
                                         minimum=1, **kwargs)

            # summary_id = list()
            # for i in range(summary_latent.size()[0]):
            #     summary_id.append(list())
            #
            # for i in range(summary_latent.size()[1]):
            #
            #     # summary, _ = self.model.z_atten(sub_z)
            #     sub_summary = summary_latent[:, i, :]
            #     init_summ_dec_state = DecState(rec_vals={"hidden": sub_summary})
            #     summ_word_ids, \
            #     summ_coll_vals = self.beamer(init_summ_dec_state,
            #                                  min_lens=min_lens,
            #                                  max_steps=max_rev_len,
            #                                  att_keys=summ_att_keys,
            #                                  att_values=summ_att_vals,
            #                                  att_mask=summ_att_mask,
            #                                  att_word_ids=summ_att_word_ids,
            #                                  minimum=1, **kwargs)
            #     for summary_l, sub_l in zip(summary_id, summ_word_ids):
            #         summary_l.extend(sub_l)
            #
            # for ii in summary_id:
            #     print(len(ii), 'cehck length -=-=-=-=--')


            # DECODING OF REVIEWS #
            z_rev_len = T.ones(z_hiddens.size()[0])
            z_rev_len = z_rev_len * z_hiddens.size()[1]
            z, _ = self.model.enc_summary(z_hiddens, z_rev_len)

            # rev_hiddens = T.transpose(rev_hiddens, 2, 1)
            # z = self.model.AvgPool(rev_hiddens)
            # z = z.squeeze(-1)



            # z = self.model.AvgPool(T.transpose(z, 2, 1))
            # z = z.squeeze(-1)

            # z_decoder = self.model.bart_model.model.decoder(input_ids=revs, encoder_hidden_states=z)
            # z_latent = z_decoder.last_hidden_state
            # _, rev_id = self.translate_sentence_beamsearch(z_latent)

        ################################################### normal decoder from here ###############################################

            rev_att_keys, \
            rev_att_vals, \
            rev_att_vals_mask = group_att_over_input_rev(inp_att_keys=att_keys,
                                                     # inp_att_vals=first_rev_hiddens,
                                                     inp_att_vals=rev_hiddens,
                                                     inp_att_mask=revs_mask,
                                                     att_indxs=other_revs,
                                                     att_indxs_mask=other_revs_mask)


            rev_att_word_ids = revs[other_revs].view(bs, -1)
            if self.min_sen_seq_len is not None:
                min_lens = [self.min_sen_seq_len] * bs
            else:
                min_lens = None

            # z, _ = self.model.z_atten(z)

            # init_rev_dec_state = DecState(rec_vals={"hidden": z_mu_q})
            init_rev_dec_state = DecState(rec_vals={"hidden": z})
            rev_word_ids, \
            rev_coll_vals = self.beamer(init_rev_dec_state,
                                        min_lens=min_lens,
                                        max_steps=max_rev_len,
                                        att_keys=rev_att_keys,
                                        att_values=rev_att_vals,
                                        att_mask=rev_att_vals_mask,
                                        att_word_ids=rev_att_word_ids,
                                        minimum=1, **kwargs)
            collect_z_id = list()

            for z_h in collect_z:
                z_rev_len = T.ones(z_h.size()[0])
                z_rev_len = z_rev_len * z_h.size()[1]
                z, _ = self.model.enc_summary(z_h, z_rev_len)
                init_rev_dec_state = DecState(rec_vals={"hidden": z})
                rev_word_ids, \
                rev_coll_vals = self.beamer(init_rev_dec_state,
                                            min_lens=min_lens,
                                            max_steps=max_rev_len,
                                            att_keys=rev_att_keys,
                                            att_values=rev_att_vals,
                                            att_mask=rev_att_vals_mask,
                                            att_word_ids=rev_att_word_ids,
                                            minimum=1, **kwargs)
                collect_z_id.append(rev_word_ids)

            return rev_coll_vals, rev_word_ids, summ_coll_vals, summ_word_ids, collect_z_id
            # return z_decoder, rev_id, summary, summary_id

    def generate_summaries(self, batch, **kwargs):
        """Generates only summaries; simplified script for inference."""
        self.model.eval()

        # batch[ModelF.REV] = T.tensor(batch[ModelF.REV], dtype=T.long)
        # print(batch[ModelF.REV], type(batch[ModelF.REV]))
        revs = batch[ModelF.REV].to(self.device)
        rev_lens = batch[ModelF.REV_LEN].to(self.device)
        revs_mask = batch[ModelF.REV_MASK].to(self.device)
        group_rev_indxs = batch[ModelF.GROUP_REV_INDXS].to(self.device)


        if ModelF.GROUP_REV_INDXS_MASK in batch:
            summ_rev_indxs_mask = batch[ModelF.GROUP_REV_INDXS_MASK].to(self.device)
        else:
            summ_rev_indxs_mask = None

        max_rev_len = revs.size(1)
        summs_nr = group_rev_indxs.size(0)

        with T.no_grad():

            # # all for bart model
            # z_output, _ = self.model.get_final_output(rev=revs)
            # z_latent = z_output.encoder_last_hidden_state
            # summary_latent = z_latent[group_rev_indxs]
            # bs = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_z = summary_latent.view(bs, group_size * seq_len, -1)
            # summary = self.model.get_final_output(input_emb=summary_z, summary=True)


            # this is for norml decoder
            # rev_embds = self.model._embds(revs) * self.model.bart_model.model.encoder.embed_scale
            # out_put = self.model.bart_model.model.encoder(input_ids=revs)
            # rev_hiddens = out_put.last_hidden_state

            rev_embds = self.model._embds(revs)
            rev_encs, rev_hiddens = self.model.encode(rev_embds, rev_lens)
            # rev_hiddens = self.model.encode(rev_embds, revs_mask)

            summary_latent = rev_encs[group_rev_indxs]
            # bs_summary = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_z = summary_latent.reshape(bs_summary, group_size * seq_len, -1)
            # summary_z = T.transpose(summary_z, 2, 1)
            summary_z = T.transpose(summary_latent, 2, 1)
            summary, _, _, _, _, collect_z = self.model.get_z(summary_z)


            # rev_code = T.transpose(rev_hiddens, 2, 1)
            # z, _, _, _, _, _ = self.model.get_z(rev_code)
            # # z, _ = self.model.z_atten(z)
            # # z = z.unsqueeze(1)
            # summary_latent = z[group_rev_indxs]
            # bs_summary = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_z = summary_latent.reshape(bs_summary, group_size * seq_len, -1)
            # summary_z = T.transpose(summary_z, 2, 1)
            # summary, _, _, _, _, collect_z = self.model.get_z(summary_z)


            # z_output, _, first_rev_hiddens = self.model.get_final_output(rev=revs)
            # rev_hiddens = z_output.encoder_last_hidden_state
            # summary_latent = rev_hiddens[group_rev_indxs]
            # bs_summary = summary_latent.size()[0]
            # group_size = summary_latent.size()[1]
            # seq_len = summary_latent.size()[2]
            # summary_z = summary_latent.view(bs_summary, group_size * seq_len, -1)
            # summary = self.model.get_final_output(input_emb=summary_z, summary=True)

            # final_rev_len = T.ones(summary.size()[0])
            # final_rev_len = final_rev_len * summary.size()[1]
            # summary, _ = self.model.enc_summary(summary, final_rev_len)

            # summary, _ = self.model.z_atten(summary)
            # decode_output = self.model.bart_model.model.decoder(inputs_embeds=summary)
            # summary = decode_output.last_hidden_state

            # summary = T.transpose(summary, 2, 1)
            # summary = self.model.AvgPool(summary)
            # summary = summary.squeeze(-1)


            ################################################### normal decoder from here ###############################################

            att_keys = self.model.create_att_keys(rev_hiddens)
            # contxt_states = self.model.get_contxt_states(rev_hiddens, rev_embds)

            summ_att_word_ids = revs[group_rev_indxs].view(summs_nr, -1)

            if self.min_sen_seq_len is not None:
                min_lens = [self.min_sen_seq_len] * summs_nr
            else:
                min_lens = None

            summ_att_keys, \
            summ_att_vals, \
            summ_att_mask = group_att_over_input(inp_att_keys=att_keys,
                                                 # inp_att_vals=first_rev_hiddens,
                                                 inp_att_vals=rev_hiddens,
                                                 inp_att_mask=revs_mask,
                                                 att_indxs=group_rev_indxs,
                                                 att_indxs_mask=summ_rev_indxs_mask)

            final_rev_len = T.ones(summary.size()[0])
            final_rev_len = final_rev_len * summary.size()[1]
            summary, _ = self.model.enc_summary(summary, final_rev_len)

            # summary, _ = self.model.z_atten(summary)

            init_summ_dec_state = DecState(rec_vals={"hidden": summary})
            summ_word_ids, \
            summ_coll_vals = self.beamer(init_summ_dec_state,
                                         min_lens=min_lens,
                                         max_steps=max_rev_len,
                                         att_keys=summ_att_keys,
                                         att_values=summ_att_vals,
                                         att_mask=summ_att_mask,
                                         att_word_ids=summ_att_word_ids,
                                         minimum=1, **kwargs)

            # summary_id = list()
            # for i in range(summary.size()[0]):
            #     summary_id.append(list())
            #
            # for i in range(summary.size()[1]):
            #
            #     # summary, _ = self.model.z_atten(sub_z)
            #     sub_summary = summary[:, i, :]
            #     init_summ_dec_state = DecState(rec_vals={"hidden": sub_summary})
            #     summ_word_ids, \
            #     summ_coll_vals = self.beamer(init_summ_dec_state,
            #                                  min_lens=min_lens,
            #                                  max_steps=max_rev_len,
            #                                  att_keys=summ_att_keys,
            #                                  att_values=summ_att_vals,
            #                                  att_mask=summ_att_mask,
            #                                  att_word_ids=summ_att_word_ids,
            #                                  minimum=1, **kwargs)
            #     for summary_l, sub_l in zip(summary_id, summ_word_ids):
            #         summary_l.extend(sub_l)
            return summ_word_ids
            # return summary_id

    def translate_sentence(self, outputs_id):

        summary_str = self.model.tokenizer.batch_decode(outputs_id, skip_special_tokens=True)

        # tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return summary_str, outputs_id

    def translate_sentence_beamsearch(self, summary_output, num_beams=3):

        batch_size = summary_output.size()[0]
        # lets run beam search using 3 beams
        # define decoder start token ids
        input_ids = T.ones((num_beams * batch_size, 1), device=self.device, dtype=T.long)
        input_ids = input_ids * self.model.bart_model.config.decoder_start_token_id

        encoder_outputs = BaseModelOutput(
            last_hidden_state=summary_output.repeat_interleave(num_beams, dim=0),
            hidden_states=None,
            attentions=None,)
        # add encoder_outputs to model keyword arguments
        model_kwargs = {"encoder_outputs": encoder_outputs}

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(batch_size=batch_size, max_length=self.model.bart_model.config.max_length,
                                       num_beams=num_beams,
                                       device=self.model.bart_model.device)
        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=200),
                MaxTimeCriteria(max_time=1),
            ]
        )

        # instantiate logits processors
        logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(3, eos_token_id=self.model.bart_model.config.eos_token_id), ])

        outputs_id = self.model.bart_model.beam_search(input_ids, beam_scorer, stopping_criteria=criteria, logits_processor=logits_processor, **model_kwargs)
        summary_str = self.model.tokenizer.batch_decode(outputs_id, skip_special_tokens=True)

        # tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return summary_str, outputs_id


    def get_longSUM(self, z, group_id, revs):
        z = z.unsqueeze(1)
        group_summary = list()
        # rev_summary = list()
        start_id = ''
        jump_flg = False
        for i in range(len(group_id)):
            group_id_i = group_id[i].split('_')[0]
            if i == 0:
                start_id = group_id_i
                test_z = z[i]
                test_rev = revs[i].unsqueeze(0)
                sub_rev = revs[i]
                continue

            if group_id_i == start_id:
                if T.equal(revs[i], sub_rev) and jump_flg is False:
                    # test_z = T.cat((test_z, z[i]), dim=0)
                    group_summary.append(test_z.unsqueeze(0))
                    # rev_summary.append(test_rev.unsqueeze(0))
                    jump_flg = True
                elif T.equal(revs[i], sub_rev) is False and jump_flg is False:
                    sub_rev = revs[i]
                    test_z = T.cat((test_z, z[i]), dim=0)
                    # test_rev = T.cat((test_rev, sub_rev.unsqueeze(0)), dim=0)
            else:
                if jump_flg is False:
                    group_summary.append(test_z.unsqueeze(0))
                jump_flg = False
                start_id = group_id_i
                test_z = z[i]
                    # test_rev = revs[i].unsqueeze(0)
            if i == len(group_id) - 1 and group_id_i == start_id and jump_flg is False:
                group_summary.append(test_z.unsqueeze(0))

        post_len_list = list()
        max_len = 0
        for zz in group_summary:
            len_zz = zz.size()[1]
            post_len_list.append(len_zz)
            if len_zz > max_len:
                max_len = len_zz
        post_len = T.FloatTensor(post_len_list)
        for i in range(len(group_summary)):
            group = group_summary[i]
            group_len = group.size()[1]
            dimension_size = group.size()[-1]
            if group_len < max_len:
                padd_zero = T.zeros((1,(max_len - group_len),dimension_size)).to(device='cuda')
                # padd_zero = T.zeros((1, (max_len - group_len), dimension_size))
                group = T.cat((group, padd_zero), dim=1)
                group_summary[i] = group

        final = T.cat(group_summary, dim=0)
        # attn_word = T.cat(rev_summary, dim=0)
        # print(attn_word.size(), 'atten word -=-=-=-=-=--=-')

        return final, post_len, max_len, post_len_list

    def construct_attn(self, final, summ_att_keys, summ_att_vals, summ_att_mask, summ_att_word_ids, summ_rev_indxs):

        num_group = summ_rev_indxs.size()[1]

        batch_size = summ_att_vals.size()[0]


        key = summ_att_keys.view(batch_size * num_group, -1, summ_att_keys.size()[-1])
        vals = summ_att_vals.view(batch_size * num_group, -1, summ_att_vals.size()[-1])
        mask = summ_att_mask.view(batch_size * num_group, -1)
        word_id = summ_att_word_ids.view(batch_size * num_group, -1)

        # batch_size = final.size()[0]
        sum_len = final.size()[1]
        batch_key_list = list()
        key_list = list()
        batch_val_list = list()
        val_list = list()
        batch_mask_list = list()
        mask_list = list()
        batch_word_id_list = list()
        word_id_list = list()

        i = 0
        while i < batch_size:
            for j in range(sum_len):
                key_list.append(key[i * num_group + j].unsqueeze(0))
                val_list.append(vals[i * num_group + j].unsqueeze(0))
                mask_list.append(mask[i * num_group + j].unsqueeze(0))
                word_id_list.append(word_id[i * num_group + j].unsqueeze(0))
            if sum_len > num_group:
                i = i + sum_len // num_group + 1
            batch_key_list.append(T.cat(key_list, dim=0).unsqueeze(0))
            batch_val_list.append(T.cat(val_list, dim=0).unsqueeze(0))
            batch_mask_list.append(T.cat(mask_list, dim=0).unsqueeze(0))
            batch_word_id_list.append(T.cat(word_id_list, dim=0).unsqueeze(0))
            key_list = []
            val_list = []
            mask_list = []
            word_id_list = []

        att_keys = T.cat(batch_key_list, dim=0)
        att_vals = T.cat(batch_val_list, dim=0)
        att_mask = T.cat(batch_mask_list, dim=0)
        att_word_ids = T.cat(batch_word_id_list, dim=0)

        bs = att_keys.size()[0]
        group_size = att_keys.size()[1]
        rev_len = att_keys.size()[2]
        att_keys = att_keys.view(bs, group_size * rev_len, -1)
        att_vals = att_vals.view(bs, group_size * rev_len, -1)
        att_mask = att_mask.view(bs, -1)
        att_word_ids = att_word_ids.view(bs, -1)

        # print(final.size(), 'final -=-=-=-=-=-=-=-=')

        # print(att_keys.size(), 'att_key =-=-=-=-=-=-=-=-=-=-=')


        return att_keys, att_vals, att_mask, att_word_ids









