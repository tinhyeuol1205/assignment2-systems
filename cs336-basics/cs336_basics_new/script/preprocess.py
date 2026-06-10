import os
import typing
from cs336_basics_new.tokenizer import Tokenizer, find_chunk_boundaries
import numpy as np
from tqdm import tqdm
import multiprocessing

def preprocess(data_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], output_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", ["<|endoftext|>"])
    num_processes = os.cpu_count() or 4
    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    worker_args = [(data_path, start, end, tokenizer) for (start, end) in tqdm(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries)-1, desc='Tokenizing chunks')]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(encode_worker, worker_args)
    with open(output_path, "wb") as fout:
        for result in results:
            arr = np.array(result, dtype=np.uint16)
            fout.write(arr.tobytes())
            fout.flush()


def encode_worker(input_path: str, start: int, end: int, tokenizer: Tokenizer):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    tokens = tokenizer.encode(chunk)
    return tokens

if __name__ == "__main__":
    preprocess("../data/data.txt", "../data/data.bin")
    

        
