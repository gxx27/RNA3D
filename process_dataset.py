import math
import pickle
from tqdm import tqdm
from transformers import T5Tokenizer
import random

def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 32768 * 32768
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)      

def random_word(sentence):
        tokens = list(sentence)
        
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15: # 15% to mask
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.additional_special_tokens[0]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = tokenizer.additional_special_tokens[random.randrange(len(tokenizer.additional_special_tokens))]

                # 10% randomly change token to current token
                else:
                    tokens[i] = token

            else:
                tokens[i] = token
                
        return tokens

if __name__ == '__main__':
    num_msa_files = 2 # can be changed here
    tokenizer = T5Tokenizer.from_pretrained('./config')

    path = './dataset/dataset1.fasta'
    length = iter_count(path)
    align_len = math.ceil(length / num_msa_files)

    data = [None] * align_len

    count = 0
    with open(path) as f:
        result = {}
        msa_mask = []
        msa_label = []
        for i, seq in tqdm(enumerate(f), total=length):
            mask = random_word(seq)
            msa_mask.append(mask[:-1])
            msa_label.append(list(seq[:-1]))
            
            if (i+1) % num_msa_files == 0:
                result['input_ids'] = msa_mask
                result['labels'] = msa_label
                data[count] = result
                msa_mask = []
                msa_label = []
                result = {}

                count += 1
                
        if result:
            data[count] = result
                
    with open('./dataset/data.pkl', 'wb') as f:
        pickle.dump(data, f)    
        
    print('process complete!')