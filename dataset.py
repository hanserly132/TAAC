"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

ENGINEERED_DENSE_FID = 100000
TIME_WINDOW_SECONDS = (3600, 6 * 3600, 86400, 3 * 86400, 7 * 86400, 30 * 86400)
RECENT_PAIR_STEPS = 50
DEFAULT_ENGINEERED_FEATURE_GROUPS = ('time', 'item')
VALID_ENGINEERED_FEATURE_GROUPS = {'time', 'item', 'pair'}


def normalize_engineered_feature_groups(groups: Optional[Any]) -> Tuple[str, ...]:
    if groups is None:
        return DEFAULT_ENGINEERED_FEATURE_GROUPS
    if isinstance(groups, str):
        groups = [g.strip() for g in groups.split(',') if g.strip()]
    normalized = tuple(dict.fromkeys(str(g).strip().lower() for g in groups if str(g).strip()))
    unknown = set(normalized) - VALID_ENGINEERED_FEATURE_GROUPS
    if unknown:
        raise ValueError(
            f"Unknown engineered feature groups: {sorted(unknown)}. "
            f"Valid values: {sorted(VALID_ENGINEERED_FEATURE_GROUPS)}")
    return normalized


def save_feature_stats(feature_stats: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(feature_stats, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_feature_stats(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _is_list_type(typ: "pa.DataType") -> bool:
    return pat.is_list(typ) or pat.is_large_list(typ)


def _log_norm(values: "npt.NDArray[np.float64]", scale: float) -> "npt.NDArray[np.float32]":
    """Log1p-normalize non-negative dense statistics to a compact range."""
    return (np.log1p(np.maximum(values, 0.0)) / np.log1p(scale)).astype(np.float32)


def _smooth_rate(count: float, positive: float, prior: float, smoothing: float) -> float:
    if count <= 0:
        return float(prior)
    return float((positive + prior * smoothing) / (count + smoothing))


def _lookup_stat(
    stats: Dict[Any, Tuple[int, int]],
    key: Any,
    prior: float,
    smoothing: float,
    current_label: int = 0,
    leave_one_out: bool = False,
) -> Tuple[float, float, float]:
    count, positive = stats.get(key, (0, 0))
    if leave_one_out and count > 0:
        count = max(count - 1, 0)
        positive = max(positive - int(current_label), 0)
    return (
        float(count),
        float(positive),
        _smooth_rate(count, positive, prior, smoothing),
    )


def _positive_values_from_pylist(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value if int(v) > 0]
    v = int(value)
    return [v] if v > 0 else []


def build_feature_stats(
    rg_info: List[Tuple[str, int, int]],
    smoothing: float = 20.0,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    """Build train-only item and item-fid target statistics.

    Validation/test batches only perform lookups against these dictionaries.
    Training batches use leave-one-out adjustment at conversion time.
    """
    item_stats: Dict[int, List[int]] = {}
    item_fid_stats: Dict[int, Dict[int, List[int]]] = {}
    global_count = 0
    global_positive = 0

    if not rg_info:
        logging.warning("Feature engineering stats requested but no train Row Groups are available.")
        return {
            'global_count': 0,
            'global_positive': 0,
            'prior': 0.0,
            'smoothing': float(smoothing),
            'item_stats': {},
            'item_fid_stats': {},
        }

    schema_names = pq.ParquetFile(rg_info[0][0]).schema_arrow.names
    item_cols = [c for c in schema_names if c.startswith('item_int_feats_')]
    columns = ['item_id', 'label_type'] + item_cols

    logging.info(
        f"Building feature-engineering stats from {len(rg_info)} train Row Groups "
        f"({len(item_cols)} item columns)")

    for file_path, rg_idx, _ in rg_info:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(
            batch_size=batch_size,
            row_groups=[rg_idx],
            columns=columns,
        ):
            names = batch.schema.names
            col_idx = {name: i for i, name in enumerate(names)}
            item_ids = batch.column(col_idx['item_id']).fill_null(0).to_numpy(
                zero_copy_only=False).astype(np.int64)
            labels = (batch.column(col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)

            for item_id, label in zip(item_ids, labels):
                if item_id <= 0:
                    continue
                rec = item_stats.setdefault(int(item_id), [0, 0])
                rec[0] += 1
                rec[1] += int(label)
            global_count += int(labels.shape[0])
            global_positive += int(labels.sum())

            for col_name in item_cols:
                fid = int(col_name.rsplit('_', 1)[1])
                fid_stats = item_fid_stats.setdefault(fid, {})
                col = batch.column(col_idx[col_name])
                if _is_list_type(col.type):
                    values_per_row = col.to_pylist()
                    for values, label in zip(values_per_row, labels):
                        for value in set(_positive_values_from_pylist(values)):
                            rec = fid_stats.setdefault(value, [0, 0])
                            rec[0] += 1
                            rec[1] += int(label)
                else:
                    arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                    arr[arr <= 0] = 0
                    for value, label in zip(arr, labels):
                        if value <= 0:
                            continue
                        rec = fid_stats.setdefault(int(value), [0, 0])
                        rec[0] += 1
                        rec[1] += int(label)

    prior = float(global_positive / global_count) if global_count > 0 else 0.0
    return {
        'global_count': int(global_count),
        'global_positive': int(global_positive),
        'prior': prior,
        'smoothing': float(smoothing),
        'item_stats': {k: (v[0], v[1]) for k, v in item_stats.items()},
        'item_fid_stats': {
            fid: {k: (v[0], v[1]) for k, v in value_stats.items()}
            for fid, value_stats in item_fid_stats.items()
        },
    }


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        enable_engineered_features: bool = True,
        feature_stats: Optional[Dict[str, Any]] = None,
        stats_leave_one_out: bool = False,
        engineered_feature_groups: Optional[Any] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            enable_engineered_features: Append hand-crafted time, item-stat,
                and sequence-item pair dense features.
            feature_stats: Train-only statistics used by engineered features.
            stats_leave_one_out: Remove the current row's label contribution
                when looking up train-built target statistics.
            engineered_feature_groups: Enabled engineered feature families.
                Defaults to ``time,item``; ``pair`` is heavier and opt-in.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.enable_engineered_features = enable_engineered_features
        self.stats_leave_one_out = stats_leave_one_out
        self.engineered_feature_groups = normalize_engineered_feature_groups(
            engineered_feature_groups)
        self.feature_stats = feature_stats or {}
        self._stat_prior = float(self.feature_stats.get('prior', 0.0))
        self._stat_smoothing = float(self.feature_stats.get('smoothing', 20.0))
        self._item_stats: Dict[int, Tuple[int, int]] = self.feature_stats.get('item_stats', {})
        self._item_fid_stats: Dict[int, Dict[int, Tuple[int, int]]] = self.feature_stats.get('item_fid_stats', {})
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        self.engineered_dense_feature_names: List[str] = []
        if self.enable_engineered_features:
            self.engineered_dense_feature_names = self._build_engineered_dense_feature_names()
            self.user_dense_schema.add(ENGINEERED_DENSE_FID, len(self.engineered_dense_feature_names))

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_engineered_dense = np.zeros(
            (B, len(self.engineered_dense_feature_names)), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def _build_engineered_dense_feature_names(self) -> List[str]:
        names: List[str] = []
        if 'time' in self.engineered_feature_groups:
            for domain in self.seq_domains:
                names.extend([
                    f'{domain}_len_log',
                    f'{domain}_recency_log',
                    f'{domain}_span_log',
                    f'{domain}_mean_gap_log',
                    f'{domain}_min_gap_log',
                    f'{domain}_max_gap_log',
                ])
                names.extend([f'{domain}_cnt_{w}s_log' for w in TIME_WINDOW_SECONDS])
            names.extend([
                'sample_hour_sin',
                'sample_hour_cos',
                'sample_dow_sin',
                'sample_dow_cos',
            ])
        if 'item' in self.engineered_feature_groups:
            names.extend([
                'item_id_count_log',
                'item_id_pos_log',
                'item_id_cvr_smooth',
            ])
            for fid, _, _ in self._item_int_cols:
                names.extend([
                    f'item_fid_{fid}_value_count_log',
                    f'item_fid_{fid}_value_pos_log',
                    f'item_fid_{fid}_value_cvr_smooth',
                ])
        if 'pair' in self.engineered_feature_groups:
            for domain in self.seq_domains:
                for fid in self.sideinfo_fids[domain]:
                    names.extend([
                        f'pair_{domain}_{fid}_hit',
                        f'pair_{domain}_{fid}_hit_count_log',
                        f'pair_{domain}_{fid}_recency_log',
                    ])
        return names

    def _compute_time_dense_features(
        self,
        timestamps: "npt.NDArray[np.int64]",
        seq_ts_by_domain: Dict[str, "npt.NDArray[np.int64]"],
        seq_lens_by_domain: Dict[str, "npt.NDArray[np.int64]"],
        B: int,
    ) -> "npt.NDArray[np.float32]":
        feats: List["npt.NDArray[np.float32]"] = []
        ts_expanded = timestamps.reshape(-1, 1)

        for domain in self.seq_domains:
            seq_ts = seq_ts_by_domain.get(domain)
            seq_lens = seq_lens_by_domain[domain].astype(np.float64)
            feats.append(_log_norm(seq_lens, 4096.0))
            if seq_ts is None or seq_ts.shape[1] == 0:
                feats.extend([np.zeros(B, dtype=np.float32) for _ in range(5 + len(TIME_WINDOW_SECONDS))])
                continue

            valid = seq_ts > 0
            time_diff = np.where(valid, np.maximum(ts_expanded - seq_ts, 0), 0)
            has_valid = valid.any(axis=1)
            large = np.full_like(time_diff, 31536000 * 2)
            recency = np.where(valid, time_diff, large).min(axis=1)
            recency = np.where(has_valid, recency, 31536000 * 2)
            max_ts = np.where(valid, seq_ts, 0).max(axis=1)
            min_ts = np.where(valid, seq_ts, np.iinfo(np.int64).max).min(axis=1)
            span = np.where(has_valid, np.maximum(max_ts - min_ts, 0), 0)

            gap_mean = np.zeros(B, dtype=np.float64)
            gap_min = np.zeros(B, dtype=np.float64)
            gap_max = np.zeros(B, dtype=np.float64)
            for i in range(B):
                vals = seq_ts[i][valid[i]]
                if vals.shape[0] < 2:
                    continue
                vals = np.sort(vals.astype(np.int64))
                gaps = np.diff(vals)
                if gaps.shape[0] == 0:
                    continue
                gap_mean[i] = float(gaps.mean())
                gap_min[i] = float(gaps.min())
                gap_max[i] = float(gaps.max())

            feats.extend([
                _log_norm(recency.astype(np.float64), 31536000.0),
                _log_norm(span.astype(np.float64), 31536000.0),
                _log_norm(gap_mean, 31536000.0),
                _log_norm(gap_min, 31536000.0),
                _log_norm(gap_max, 31536000.0),
            ])
            for window in TIME_WINDOW_SECONDS:
                cnt = ((time_diff <= window) & valid).sum(axis=1).astype(np.float64)
                feats.append(_log_norm(cnt, 4096.0))

        hour = ((timestamps // 3600) % 24).astype(np.float64)
        dow = ((timestamps // 86400 + 4) % 7).astype(np.float64)
        feats.extend([
            np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32),
            np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32),
            np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32),
            np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32),
        ])
        return np.stack(feats, axis=1).astype(np.float32)

    def _compute_item_stat_features(
        self,
        item_ids: "npt.NDArray[np.int64]",
        labels: "npt.NDArray[np.int64]",
        item_value_by_fid: Dict[int, "npt.NDArray[np.int64]"],
        B: int,
    ) -> "npt.NDArray[np.float32]":
        feats: List["npt.NDArray[np.float32]"] = []
        item_count = np.zeros(B, dtype=np.float64)
        item_pos = np.zeros(B, dtype=np.float64)
        item_rate = np.full(B, self._stat_prior, dtype=np.float32)
        for i, item_id in enumerate(item_ids):
            count, positive, rate = _lookup_stat(
                self._item_stats,
                int(item_id),
                self._stat_prior,
                self._stat_smoothing,
                int(labels[i]),
                self.stats_leave_one_out,
            )
            item_count[i] = count
            item_pos[i] = positive
            item_rate[i] = rate
        feats.extend([
            _log_norm(item_count, 1_000_000.0),
            _log_norm(item_pos, 100_000.0),
            item_rate.astype(np.float32),
        ])

        for fid, _, _ in self._item_int_cols:
            value_stats = self._item_fid_stats.get(fid, {})
            values = item_value_by_fid.get(fid, np.zeros(B, dtype=np.int64))
            count_arr = np.zeros(B, dtype=np.float64)
            pos_arr = np.zeros(B, dtype=np.float64)
            rate_arr = np.full(B, self._stat_prior, dtype=np.float32)
            for i, value in enumerate(values):
                if value <= 0:
                    continue
                count, positive, rate = _lookup_stat(
                    value_stats,
                    int(value),
                    self._stat_prior,
                    self._stat_smoothing,
                    int(labels[i]),
                    self.stats_leave_one_out,
                )
                count_arr[i] = count
                pos_arr[i] = positive
                rate_arr[i] = rate
            feats.extend([
                _log_norm(count_arr, 1_000_000.0),
                _log_norm(pos_arr, 100_000.0),
                rate_arr.astype(np.float32),
            ])
        return np.stack(feats, axis=1).astype(np.float32)

    def _compute_pair_dense_features(
        self,
        item_ids: "npt.NDArray[np.int64]",
        item_value_by_fid: Dict[int, "npt.NDArray[np.int64]"],
        seq_side_values: Dict[str, Dict[int, "npt.NDArray[np.int64]"]],
        seq_ts_by_domain: Dict[str, "npt.NDArray[np.int64]"],
        timestamps: "npt.NDArray[np.int64]",
        B: int,
    ) -> "npt.NDArray[np.float32]":
        feats: List["npt.NDArray[np.float32]"] = []
        for domain in self.seq_domains:
            seq_ts = seq_ts_by_domain.get(domain)
            for fid in self.sideinfo_fids[domain]:
                seq_vals = seq_side_values[domain][fid]
                item_vals = item_value_by_fid.get(fid)
                hit = np.zeros(B, dtype=np.float32)
                hit_count = np.zeros(B, dtype=np.float64)
                hit_recency = np.zeros(B, dtype=np.float64)
                for i in range(B):
                    candidates = [int(item_ids[i])]
                    for values in item_value_by_fid.values():
                        value = int(values[i])
                        if value > 0:
                            candidates.append(value)
                    if item_vals is not None and int(item_vals[i]) > 0:
                        candidates.append(int(item_vals[i]))
                    candidates = list({v for v in candidates if v > 0})
                    if not candidates:
                        continue
                    valid_positions = np.where(np.isin(seq_vals[i, :RECENT_PAIR_STEPS], candidates))[0]
                    if valid_positions.shape[0] == 0:
                        continue
                    hit[i] = 1.0
                    hit_count[i] = float(valid_positions.shape[0])
                    if seq_ts is not None and seq_ts.shape[1] > 0:
                        matched_ts = seq_ts[i, valid_positions]
                        matched_ts = matched_ts[matched_ts > 0]
                        if matched_ts.shape[0] > 0:
                            hit_recency[i] = float(max(int(timestamps[i]) - int(matched_ts.max()), 0))
                feats.extend([
                    hit,
                    _log_norm(hit_count, float(RECENT_PAIR_STEPS)),
                    _log_norm(hit_recency, 31536000.0),
                ])
        return np.stack(feats, axis=1).astype(np.float32)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        arrow_col = arrow_col.fill_null([])
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        arrow_col = arrow_col.fill_null([])
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()
        item_ids = batch.column(self._col_idx['item_id']).fill_null(0).to_numpy(
            zero_copy_only=False).astype(np.int64)
        item_ids[item_ids <= 0] = 0

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        item_value_by_fid: Dict[int, "npt.NDArray[np.int64]"] = {}
        for (fid, _, _), (ci, dim, offset, vs) in zip(self._item_int_cols, self._item_int_plan):
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
                item_value_by_fid[fid] = arr.copy()
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded
                item_value_by_fid[fid] = padded[:, 0].copy() if dim > 0 else np.zeros(B, dtype=np.int64)

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded
        engineered_offset = self.user_dense_schema.total_dim - len(self.engineered_dense_feature_names)

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        seq_ts_by_domain: Dict[str, "npt.NDArray[np.int64]"] = {}
        seq_lens_by_domain: Dict[str, "npt.NDArray[np.int64]"] = {}
        seq_side_values: Dict[str, Dict[int, "npt.NDArray[np.int64]"]] = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]
            sideinfo_fids = self.sideinfo_fids[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col = col.fill_null([])
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())
            seq_lens_by_domain[domain] = lengths.copy()
            seq_side_values[domain] = {
                fid: out[:, slot, :].copy()
                for slot, fid in enumerate(sideinfo_fids)
            }

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            ts_padded = np.zeros((B, max_len), dtype=np.int64)
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_col = ts_col.fill_null([])
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            seq_ts_by_domain[domain] = ts_padded.copy()

        if self.enable_engineered_features and self.engineered_dense_feature_names:
            engineered_dense = self._buf_engineered_dense[:B]
            engineered_dense[:] = 0
            parts = []
            if 'time' in self.engineered_feature_groups:
                parts.append(self._compute_time_dense_features(
                    timestamps, seq_ts_by_domain, seq_lens_by_domain, B))
            if 'item' in self.engineered_feature_groups:
                parts.append(self._compute_item_stat_features(
                    item_ids, labels, item_value_by_fid, B))
            if 'pair' in self.engineered_feature_groups:
                parts.append(self._compute_pair_dense_features(
                    item_ids, item_value_by_fid, seq_side_values, seq_ts_by_domain, timestamps, B))
            engineered_dense[:, :] = np.concatenate(parts, axis=1)
            user_dense[:, engineered_offset:self.user_dense_schema.total_dim] = engineered_dense

        result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    enable_engineered_features: bool = True,
    stats_smoothing: float = 20.0,
    engineered_feature_groups: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)
    engineered_feature_groups = normalize_engineered_feature_groups(
        engineered_feature_groups)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    if valid_ratio <= 0 or total_rgs <= 1:
        n_valid_rgs = 0
    else:
        n_valid_rgs = min(total_rgs - 1, max(1, int(total_rgs * valid_ratio)))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])
    train_rg_info = rg_info[:n_train_rgs]

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    feature_stats = None
    if enable_engineered_features and 'item' in engineered_feature_groups:
        feature_stats = build_feature_stats(
            train_rg_info,
            smoothing=stats_smoothing,
            batch_size=max(batch_size, 4096),
        )
        logging.info(
            "Feature-engineering stats: rows=%s, positives=%s, prior=%.6f",
            feature_stats.get('global_count'),
            feature_stats.get('global_positive'),
            feature_stats.get('prior', 0.0),
        )

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        enable_engineered_features=enable_engineered_features,
        feature_stats=feature_stats,
        stats_leave_one_out=True,
        engineered_feature_groups=engineered_feature_groups,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs) if n_valid_rgs > 0 else (total_rgs, total_rgs),
        clip_vocab=clip_vocab,
        enable_engineered_features=enable_engineered_features,
        feature_stats=feature_stats,
        stats_leave_one_out=False,
        engineered_feature_groups=engineered_feature_groups,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")
    if enable_engineered_features:
        logging.info(
            "Engineered dense features appended to user_dense: %s dims, groups=%s",
            len(train_dataset.engineered_dense_feature_names),
            ','.join(engineered_feature_groups),
        )

    return train_loader, valid_loader, train_dataset
