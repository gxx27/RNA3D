from tqdm import tqdm
from handle import read_fasta
import pickle
import json
import glob
import os

def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 32768 * 32768
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)   

if __name__ == '__main__':
    num_msa_files = 30 # can be changed here

    msa_path = './msa_folder'
    path_list = glob.glob(msa_path+'/*.alnfasta')
    fasta_path_list = []
    
    # data clean
    for path in path_list :
        fasta_path = path.replace('.alnfasta', '.fasta')
        fasta_path_list.append(fasta_path)
        if os.path.exists(fasta_path):
            print(f'file {fasta_path} exist!')
            continue
        
        sequences = []
        for _, seq in read_fasta(path):
            sequences.append(seq)
        sequences = list(set(sequences))
        sequences = [s for s in sequences if s != '-' * len(s)]
        
        with open(fasta_path, 'w') as file:
            for i, seq in enumerate(sequences, start=1):
                file.write(f'{seq}\n')
        print(f'file {path} cleaned success!')
    
    
    m = 15
    data = []
    for i, path in tqdm(enumerate(fasta_path_list), total=len(fasta_path_list)):
        with open(path) as f:
            sequences = []
            result = {}
            for j, seq in enumerate(f): # f应该是某个msa
                sequences.append(list(seq[:-1]))
                if (j+1) % num_msa_files == 0:
                    src = sequences[:m]
                    tgt = sequences[m:]
                
                    result['input_ids'] = src
                    result['labels'] = tgt

                    data.append(result)

                    sequences.clear()
                    result = {}
     
    with open(msa_path+'/data.pkl', 'wb') as f:
        pickle.dump(data, f)    
    # with open(msa_path+'/data.json', 'w') as f:
    #     json.dump(data, f)
        
    print('process complete!')