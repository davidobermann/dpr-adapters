import enum
import os
import sys
from abc import ABC
from typing import Union, Tuple

from torch._C import dtype
from transformers import BertPreTrainedModel, PfeifferConfig, DistilBertAdapterModel, ParallelConfig, AdapterSetup, \
    ModelWithFlexibleHeadsAdaptersMixin, BertModel, AutoAdapterModel, PreTrainedModel, BertConfig
from transformers.adapters import PredictionHead, AdapterConfig, MultiLabelClassificationHead, \
    ClassificationHead, MultipleChoiceHead, TaggingHead, QuestionAnsweringHead, BiaffineParsingHead, \
    BertStyleMaskedLMHead, CausalLMHead
from transformers.adapters.model_mixin import EmbeddingAdaptersWrapperMixin
import transformers.adapters.composition as ac

sys.path += ['./']
import torch
from torch import nn
import transformers
import torch.nn.functional as F
from torch.cuda.amp import autocast


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from BertModel to use from_pretrained
    """

    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class BaseModelDot(EmbeddingMixin):
    def _text_encode(self, input_ids, attention_mask):
        # TODO should raise NotImplementedError
        # temporarily do this  
        return None

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids,
                                     attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)


class DPRHead(PredictionHead):
    def __init__(self, model, head_name, **kwargs):
        super().__init__(head_name)

        config = model.config

        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size

        self.hiddensize = config.hidden_size

        self.embeddingHead = nn.Linear(self.hiddensize, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)

    def first(self, emb_all):
        return emb_all[0][:, 0]

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        full_emb = self.first(outputs)
        head_output = self.norm(self.embeddingHead(full_emb))
        return head_output


class BertAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # rewritten class to remove pooling layer
        self.bert = BertModel(config, add_pooling_layer=False)

        self._init_head_modules()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        **kwargs
    ):
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # BERT & RoBERTa return the pooled output as second item, we don't need that in these heads
        if not return_dict:
            head_inputs = (outputs[0],) + outputs[2:]
        else:
            head_inputs = outputs
        pooled_output = outputs[1]

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                head_inputs,
                head_name=head,
                attention_mask=attention_mask,
                return_dict=return_dict,
                pooled_output=pooled_output,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs

    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
        "tagging": TaggingHead,
        "multiple_choice": MultipleChoiceHead,
        "question_answering": QuestionAnsweringHead,
        "dependency_parsing": BiaffineParsingHead,
        "masked_lm": BertStyleMaskedLMHead,
        "causal_lm": CausalLMHead,
    }


class AdapterBertDot(BaseModelDot, BertAdapterModel):
    def __init__(self, config, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        BertAdapterModel.__init__(self, config)
        if int(transformers.__version__[0]) == 4:
            config.return_dict = False
        self.bert = BertModel(config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)

        self.task_name = 'dpr'

    def load_my_adapter(self, adapter_path):
        return self.load_adapter(adapter_path)

    def init_adapter_setup(self, config):
        self.add_adapter(self.task_name, config=config)

    def first(self, emb_all):
        return emb_all[0][:, 0]

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.first(outputs1)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids, attention_mask=attention_mask)
        query1 = outputs1
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)


class CompoundModel(nn.Module):
    def __init__(self, config, init_path):
        super().__init__()

        self.bertQ = BertModel.from_pretrained(init_path, config=config)
        self.bertD = BertModel.from_pretrained(init_path, config=config)

        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size

    def _init_weights(self, module):
        self.bertQ.init_weights()
        self.bertD.init_weights()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        self.bertQ.resize_position_embeddings(new_num_position_embeddings)
        self.bertD.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        return self.bertQ.get_position_embeddings(), self.bertD.get_position_embeddings()

    def _reorder_cache(self, past, beam_idx):
        pass

    def init_adapter_setup(self, config):
        self.bertQ.add_adapter(self.task_name + 'Q', config)
        self.bertD.add_adapter(self.task_name + 'D', config)

        self.bertQ.freeze_model(freeze=True)
        self.bertD.freeze_model(freeze=True)

        self.bertQ.train_adapter([self.task_name + 'Q'])
        self.bertD.train_adapter([self.task_name + 'D'])

        print(self.bertQ.adapter_summary())
        print(self.bertD.adapter_summary())

    def save_all_adapters(self, output_dir):
        self.bertQ.save_all_adapters(output_dir)
        self.bertD.save_all_adapters(output_dir)

    def load_adapters(self, adapter_path):
        self.bertQ.freeze_model(freeze=True)
        self.bertD.freeze_model(freeze=True)

        print('loading: ' + adapter_path + '/dprQ')
        print('loading: ' + adapter_path + '/dprD')

        nameQ = self.bertQ.load_adapter(adapter_path + '/dprQ')
        nameD = self.bertD.load_adapter(adapter_path + '/dprD')

        self.bertQ.set_active_adapters([nameQ])
        self.bertD.set_active_adapters([nameD])

        self.bertQ.train_adapter([self.task_name + 'Q'])
        self.bertD.train_adapter([self.task_name + 'D'])

        print(self.bertD.adapter_summary())
        print(self.bertQ.adapter_summary())
        return nameQ, nameD

    def first(self, emb_all):
        return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.bertQ(input_ids=input_ids, attention_mask=attention_mask)
        query1 = self.first(outputs1)
        return query1

    def body_emb(self, input_ids, attention_mask):
        outputs1 = self.bertD(input_ids=input_ids, attention_mask=attention_mask)
        body = self.first(outputs1)
        return body

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)



class AdapterBertDot_dual(BaseModelDot, BertAdapterModel):
    def __init__(self, config, model_argobj=None, adapter_path=None):
        BaseModelDot.__init__(self, model_argobj)
        BertAdapterModel.__init__(self, config)
        if int(transformers.__version__[0]) == 4:
            config.return_dict = False
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)

        self.task_name = 'dpr'
        # --adapter_path ./data/1000/dpr\

    def load_my_adapters(self, adapter_pathQ, adapter_pathD):
        name1 = self.load_adapter(adapter_pathQ)
        name2 = self.load_adapter(adapter_pathD)
        return name1, name2

    def enable_training(self):
        self.freeze_model(freeze=True)
        self.train_adapter([self.task_name + 'D'])
        self.train_adapter([self.task_name + 'Q'])

        for (n, p) in self.named_parameters():
            #if 'adapter' in n and p.requires_grad == True:
            #    print(n, 'before training after loading in model function')
            #    print(p.mean().item())
            print(n, p.requires_grad)

    def init_adapter_setup(self, config):
        self.add_adapter(self.task_name + 'Q', config)
        self.add_adapter(self.task_name + 'D', config)

        self.freeze_model(freeze=True)

        self.train_adapter([self.task_name + 'Q'])
        self.train_adapter([self.task_name + 'D'])

        self.active_adapters = ac.Split(self.task_name + 'Q', self.task_name + 'D', split_index=24)

        print(self.adapter_summary())

    def first(self, emb_all):
        return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        return self.first(self.bertQ(input_ids=input_ids, attention_mask=attention_mask))

    def body_emb(self, input_ids, attention_mask):
        return self.first(self.bertD(input_ids=input_ids, attention_mask=attention_mask))

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)

class CompoundModel_InBatch(CompoundModel):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             rel_pair_mask, hard_pair_mask)

class AdapterBertDot_dual_InBatch(AdapterBertDot_dual):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             rel_pair_mask, hard_pair_mask)

class AdapterBertDot_InBatch(AdapterBertDot):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             rel_pair_mask, hard_pair_mask)


class AdapterBertDot_Rand(AdapterBertDot):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return randneg_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             hard_pair_mask)


def inbatch_train(query_encode_func, doc_encode_func,
                  input_query_ids, query_attention_mask,
                  input_doc_ids, doc_attention_mask,
                  other_doc_ids=None, other_doc_attention_mask=None,
                  rel_pair_mask=None, hard_pair_mask=None):
    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)

    batch_size = query_embs.shape[0]
    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)
        # print("batch_scores", batch_scores)
        single_positive_scores = torch.diagonal(batch_scores, 0)
        # print("positive_scores", positive_scores)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)
        if rel_pair_mask is None:
            rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)
            # print("mask", mask)
        batch_scores = batch_scores.reshape(-1)
        logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                  batch_scores.unsqueeze(1)], dim=1)
        # print(logit_matrix)
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)
        # print(loss)
        # print("\n")
        first_loss, first_num = loss.sum(), rel_pair_mask.sum()

    if other_doc_ids is None:
        return (first_loss / first_num,)

    # other_doc_ids: batch size, per query doc, length
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)

    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                        other_batch_scores.unsqueeze(1)], dim=1)
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
        # print(loss)
        # print("\n")
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)

    return ((first_loss + second_loss) / (first_num + second_num),)


def randneg_train(query_encode_func, doc_encode_func,
                  input_query_ids, query_attention_mask,
                  input_doc_ids, doc_attention_mask,
                  other_doc_ids=None, other_doc_attention_mask=None,
                  hard_pair_mask=None):
    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)

    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)
        single_positive_scores = torch.diagonal(batch_scores, 0)
    # other_doc_ids: batch size, per query doc, length
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)

    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
        other_batch_scores = other_batch_scores.reshape(-1)
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                        other_batch_scores.unsqueeze(1)], dim=1)
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)
        other_loss = -1.0 * other_lsm[:, 0]
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            second_loss, second_num = other_loss.sum(), len(other_loss)
    return (second_loss / second_num,)
