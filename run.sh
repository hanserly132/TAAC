#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --num_epochs 10 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 4 \
    --batch_size  512\
    --seq_max_lens "seq_a:512,seq_b:512,seq_c:1024,seq_d:1024" \
    "$@"