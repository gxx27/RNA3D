import re
import argparse
import torch
import random
import math
from typing import Sequence, Tuple, List, Union
from torch.utils.data import Dataset

RawMSA = Sequence[Tuple[str, str]]

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for RNA Augmentor")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()
    return args

def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result


def read_alignment_lines(
    lines,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip().lstrip(">")
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)

class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        # batch_labels, seq_str_list = zip(*raw_batch)
        # seq_str_list = zip(*raw_batch)
        seq_str_list = raw_batch
        seq_encoded_list = [self.tokenizer(self._tokenize(seq_str)).input_ids for seq_str in seq_str_list] # 向量化 'A U G C' -> [220, 115, 56, 89]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len 
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.tokenizer.pad_token_id)
        # labels = []
        strs = []
        # for i, (label, seq_str, seq_encoded) in enumerate(
        #     zip(batch_labels, seq_str_list, seq_encoded_list)
        # ):
        for i, (seq_str, seq_encoded) in enumerate(
            zip(seq_str_list, seq_encoded_list)
        ):
            # labels.append(label)
            strs.append(seq_str)
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i,0:len(seq_encoded)] = seq

        # return labels, strs, tokens
        return strs, tokens
    def _tokenize(self,sequence): # 将序列字符串进行分词处理 将字符串转换为由空格分隔的字符列表
        return ' '.join(list(sequence)) # 'AUGC' -> 'A U G C'
    
    
class DataCollatorForMSA(BatchConverter):
    def msa_batch_convert(self, inputs: Union[Sequence[RawMSA], RawMSA]): # 输入可以是一个序列 包含多个RawMSA对象 或者单个RawMSA对象 每个RawMSA是一个tuple 包含标签和氨基酸序列
        # RawMSA: Sequence[label:str,acid_seq:str]
        if isinstance(inputs[0][0], str): # 判断是不是单个RawMSA对象
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore
        batch_size = len(raw_batch) # 20
        max_alignments = max(len(msa) for msa in raw_batch) # 3
        # max_seqlen = max(len(msa[0]) for msa in raw_batch) # 213

        max_seqlen = max(len(seq) for msa in raw_batch for seq in msa)

        for msa in raw_batch: # padding
            for seq in msa:
                seq += [self.tokenizer.pad_token] * (max_seqlen - len(seq))

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen+1,
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.tokenizer.pad_token_id)
        # labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            # msa_seqlens = set(len(seq) for _, seq in msa) # msa序列的长度
            msa_seqlens = set(len(seq) for seq in msa) # 213
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            # msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            msa_strs, msa_tokens = super().__call__(msa)
            
            # labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return tokens

    def __call__(self, batch): # example可以是一个dict 
        input_ids = self.msa_batch_convert([example["input_ids"] for example in batch]) # convert source 原序列
        labels = self.msa_batch_convert([example["labels"] for example in batch]) # convert target 目标序列
        labels[labels==self.tokenizer.pad_token_id]=-100 # 计算损失时忽略填充部分
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).type_as(input_ids) # input_ids中不等于填充标记的地方作为encoder注意力掩码
        decoder_attention_mask = labels.ne(self.tokenizer.pad_token_id).type_as(input_ids) # labels中不等于填充标记的地方作为decoder注意力掩码
        return {'input_ids':input_ids,'labels':labels,"attention_mask":attention_mask,"decoder_attention_mask":decoder_attention_mask}

class MSADataSet(Dataset):
    def __init__(self, data_args, num_msa_files):
        super().__init__()
        self.data_args = data_args
        self.data_path = self.data_args.train_file
        self.num_msa_files = num_msa_files
        
        self.tokenizer = data_args.tokenizer
        
        # self.max_seq_len = 0
        self.length = iter_count(self.data_path) # 得到数据集长度
        align_len = math.ceil(self.length / self.num_msa_files)
        self.lines = [None] * align_len
        align = []
        count = 0
        with open(self.data_path) as f:
            for i, line in enumerate(f):                
                align.append(line)
    
                if (i+1) % self.num_msa_files == 0:
                    self.lines[count] = align
                    align = []
                    count += 1
                    
            if align:
                self.lines[count] = align
        # self.data = []      
        # for msa in self.lines:
        #     result = {}
        #     msa_mask = []
        #     msa_label = []
        #     for seq in msa:
        #         mask = self.random_word(seq)
        #         msa_mask.append(mask[:-1])
        #         msa_label.append(list(seq[:-1]))

        #     result['input_ids'] = msa_mask
        #     result['labels'] = msa_label
        #     self.data.append(result)

    def __len__(self):
        # return len(self.lines)
        return self.length

    def __getitem__(self, index):
        # return self.data[index]
        sequence = {}
        msa = self.lines[index]
        msa_mask = []
        msa_label = []
        
        # max_seqlen = max(len(msa[0]) for msa in raw_batch)
        
        for seq in msa:
            mask = self.random_word(seq)
            
            mask = mask[:-1]
            seq = list(seq[:-1])
            
            # if len(mask) < max_length:
            #     mask += [self.tokenizer.pad_token] * (max_length - len(mask))
            #     seq += [self.tokenizer.pad_token] * (max_length - len(seq))
            
            msa_mask.append(mask)
            msa_label.append(seq)

        sequence['input_ids'] = msa_mask
        sequence['labels'] = msa_label
        # return self.data[index]
        return sequence
        
    def random_word(self, sentence):
        # tokens = sentence.split()
        tokens = list(sentence)
        # output_label = []
        
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15: # 15% to mask
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # tokens[i] = self.vocab.itos[self.vocab.mask_index] # '<mask>'
                    tokens[i] = self.tokenizer.additional_special_tokens[0]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # tokens[i] = self.vocab.itos[random.randrange(len(self.vocab))] # 随便选一个
                    tokens[i] = self.tokenizer.additional_special_tokens[random.randrange(len(self.tokenizer.additional_special_tokens))]

                # 10% randomly change token to current token
                else:
                    # tokens[i] = self.vocab.itos[self.vocab.stoi.get(token, self.vocab.unk_index)] # 当前的
                    pass
                # output_label.append(self.vocab.itos[self.vocab.stoi.get(token, self.vocab.unk_index)])

            else: # 不进行掩码操作
                # tokens[i] = self.vocab.itos[self.vocab.stoi.get(token, self.vocab.unk_index)]
                pass
                # output_label.append(self.vocab.itos[0])

        # return tokens, output_label
        return tokens
    
def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 32768 * 32768
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)      