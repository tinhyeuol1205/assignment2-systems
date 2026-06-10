import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import multiprocessing
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Hàm chính điều phối toàn bộ quá trình:
    1. Đọc dữ liệu từ file.
    2. Khởi tạo vocab cơ bản (256 bytes) và quản lý special_tokens.
    3. Tiền xử lý dữ liệu (dùng helper function).
    4. Vòng lặp: Tìm cặp token phổ biến nhất -> Gộp chúng lại -> Cập nhật vocab & merges.
    5. Trả về kết quả cuối cùng.
    """
    vocab = init_base_vocab(special_tokens)
    merges = []
    next_id = 256 + len(special_tokens)
    word_counts = get_word_frequencies(input_path, special_tokens)
    words = [list(w) for w in word_counts.keys()]
    counts = list(word_counts.values())
    where_is_pair = defaultdict(set)

    # Count pair frequencies
    pair_counts = defaultdict(int)
    for idx, (token, count) in enumerate(word_counts.items()):
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_counts[pair] += count
            where_is_pair[pair].add(idx)

    for i in tqdm(range(vocab_size - len(vocab)), desc='Training BPE'):
        pair_to_merge = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        indices_to_update = where_is_pair[pair_to_merge]
        new_token = pair_to_merge[0] + pair_to_merge[1]
        vocab[next_id + i] = new_token
        merges.append(pair_to_merge)
        
        for idx in indices_to_update:
            word = words[idx]
            count = counts[idx]
            ids = 0
            while ids < len(word) - 1:
                if (word[ids], word[ids + 1]) == pair_to_merge:
                    if ids > 0:
                        pair_counts[word[ids - 1], word[ids]] -= count
                        pair_counts[word[ids - 1], new_token] += count
                        where_is_pair[word[ids - 1], new_token].add(idx)
                    if ids < len(word) - 2:
                        pair_counts[word[ids + 1], word[ids + 2]] -= count
                        pair_counts[new_token, word[ids + 2]] += count                    
                        where_is_pair[new_token, word[ids + 2]].add(idx)
                    word[ids:ids+2] = [new_token]

                ids += 1
        del pair_counts[pair_to_merge]
        del where_is_pair[pair_to_merge]            

    return (vocab, merges)

def init_base_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    Nhiệm vụ: Tạo từ điển vocab ban đầu.
    - Map các ID từ 0-255 với 256 giá trị bytes cơ bản (0x00 đến 0xFF).
    - Map các ID tiếp theo (256, 257...) cho các chuỗi trong special_tokens.
    """
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])
    
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")
    
    return vocab


def get_word_frequencies(
    input_path: str | os.PathLike, 
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    Nhiệm vụ: Đọc corpus, tiền xử lý và lập thống kê từ vựng.
    - Tích hợp logic chia chunk file từ `pretokenization_example.py`.
    - Phân tách chunk text dựa vào `special_tokens`.
    - Dùng GPT-2 Regex để cắt thành các từ.
    - Convert mỗi từ thành 1 tuple các bytes riêng lẻ.
    - Trả về Dictionary thống kê. Ví dụ: {(b'h', b'e', b'l', b'l', b'o'): 150}
    """

    word_counts = defaultdict(int)
    num_processes = os.cpu_count() or 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
    worker_args = [(input_path, start, end, special_tokens) for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc='Finding chunk boundaries', total=len(boundaries)-1)]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(get_word_frequencies_worker, worker_args)
        for result in results:
            for token, count in result.items():
                word_counts[token] += count

    return word_counts

def get_word_frequencies_worker(input_path, start, end, special_tokens):
    word_counts = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        texts = re.split(pattern, chunk)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # Match against the regex pattern
        for text in texts:
            matches = re.finditer(PAT, text)
            tokens = [tuple(bytes([b]) for b in m.group(0).encode("utf-8")) for m in matches]
            for token in tokens:
                word_counts[token] += 1
    
    return word_counts



    