import sys
from abc import ABC, abstractmethod

from transformers import BertModel
from transformers.adapters import PredictionHead

sys.path += ['./']
import torch
from torch import nn
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
        outputs1 = self._text_encode(input_ids=input_ids, attention_mask=attention_mask)
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


class Saveable(ABC):
    @abstractmethod
    def save_all_adapters(self, output_dir):
        pass

    @abstractmethod
    def save_all_adapter_fusions(self, output_dir):
        pass

    @abstractmethod
    def save_all_heads(self, output_dir):
        pass


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

class BertDot_SingleAdapter(nn.Module, Saveable, BaseModelDot):
    def __init__(self, config, init_path):
        super().__init__()

        self.bert = BertModel.from_pretrained(init_path, config=config)

        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size

        self.task_name = 'dpr'

    def load_adapter(self, adapter_path):
        name = self.bert.load_adapter(adapter_path)
        self.bert.set_active_adapters([name])
        print(self.bert.adapter_summary())

    def init_adapter_setup(self, config):
        self.bert.add_adapter(self.task_name, config=config)
        self.bert.freeze_model(freeze=True)
        self.bert.train_adapter([self.task_name])
        print(self.bert.adapter_summary())

    def save_all_adapters(self, output_dir):
        self.bert.save_all_adapters(output_dir)

    def save_all_adapter_fusions(self, output_dir):
        pass

    def save_all_heads(self, output_dir):
        pass

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

class BertDot_DualAdapter(nn.Module, BaseModelDot, Saveable):
    def __init__(self, config, init_path):
        super().__init__()

        self.bertQ = BertModel.from_pretrained(init_path, config=config)
        self.bertD = BertModel.from_pretrained(init_path, config=config)

        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size

        self.task_name = 'dpr'

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

    def save_all_adapter_fusions(self, output_dir):
        pass

    def save_all_heads(self, output_dir):
        pass

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



class BertDot_DualAdapter_InBatch(BertDot_DualAdapter):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             rel_pair_mask, hard_pair_mask)


class BertDot_SingleAdapter_InBatch(BertDot_SingleAdapter):
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return inbatch_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             rel_pair_mask, hard_pair_mask)


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