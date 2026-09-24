# Dataset

The presets train on **BookCorpus** (a subset of [The Pile](https://pile.eleuther.ai/)),
tokenized with the GPT-2 BPE vocab into Megatron's indexed `.bin` / `.idx` mmap format.

The dataset is **not** vendored in this repo. Place the four files below under
`Megatron-DeepSpeed/dataset/`, which is where `examples/gpt.sh` expects them:

```
Megatron-DeepSpeed/dataset/
├── gpt2-vocab.json
├── gpt2-merges.txt
├── BookCorpusDataset_text_document.bin
└── BookCorpusDataset_text_document.idx
```

Megatron-DeepSpeed ships helper scripts for all four in `dataset/`.

## 1. Vocab + merges

```bash
cd Megatron-DeepSpeed/dataset
bash download_vocab.sh
```

This fetches `gpt2-vocab.json` and `gpt2-merges.txt`.

## 2. Pre-tokenized BookCorpus

Upstream ships `download_books.sh`, which is meant to pull the **already-preprocessed**
`.bin` / `.idx` directly (no tokenization step needed):

```bash
cd Megatron-DeepSpeed/dataset
bash download_books.sh
```

> **Heads up:** the download links in `download_books.sh` are no longer live, so this
> step currently fails — see
> [deepspeedai/Megatron-DeepSpeed#233](https://github.com/deepspeedai/Megatron-DeepSpeed/issues/233).
> The corpus is a BookCorpus subset of The Pile (<https://pile.eleuther.ai/>). If your
> site keeps a shared preprocessed copy, copy or symlink its
> `BookCorpusDataset_text_document.{bin,idx}` into `dataset/`; otherwise preprocess your
> own corpus as in step 3.

## 3. (Optional) Preprocess your own corpus

To train on different data, tokenize a newline-delimited JSON file (one document per
line with a `"text"` field) from the `Megatron-DeepSpeed` root:

```bash
python tools/preprocess_data.py \
    --input dataset/my_corpus.jsonl \
    --output-prefix dataset/MyDataset \
    --vocab-file dataset/gpt2-vocab.json \
    --merge-file dataset/gpt2-merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --json-keys text \
    --workers 16
```

This produces `dataset/MyDataset_text_document.{bin,idx}`. Point `DATA_PATH` in
`examples/gpt.sh` at the new `--output-prefix` (`dataset/MyDataset_text_document`).
