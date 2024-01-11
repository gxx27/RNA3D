import re
from tqdm import tqdm

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
        elif len(line) > 0 and line[0] == "#":
            pass
        elif len(line) > 0 and line[0] == "\n":
            pass
        else:
            try:
                assert isinstance(seq, str)
            except Exception:
                import ipdb
                ipdb.set_trace()
            line = line.replace("\x00", "")
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)

if __name__ == '__main__':
    data_path = './data.fasta'
    
    sequences = []
    max_length = 1000
    for _, seq in tqdm(read_fasta(path=data_path), total=max_length):
        sequences.append(seq)
        sequences = list(set(sequences))
        if len(sequences) >= max_length:
            break
    
    
    with open('dataset.fasta', 'w') as file:
        for i, seq in enumerate(sequences, start=1):
            file.write(f'{seq}\n')

    print('Sequences saving finished.')