# MIT License
# 
# Copyright (c) 2018-2019 Stichting SingularityNET
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Reputation Service API, including Rating Service and Ranking Service
import pandas as pd
import numpy as np

# Must be imported (as requested).
try:
    import reputation_calculation  # noqa: F401
except ImportError as e:
    raise ImportError("Missing required module 'reputation_calculation'. "
        "Ensure it is available on PYTHONPATH or in the project directory.") from e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

"""
Reputation Generic Service interface definition
"""        
class LiquidRankConfig:
    """
    Configuration container for Liquid Rank reputation.

    All parameters have defaults.
    If the user does not explicitly pass a parameter,
    the default defined here is used.
    """

    def __init__(
        self,
        # Periodization
        period: PeriodMode = "D",          # Days by default
        start_ts: Optional[int] = None,    # unix seconds
        end_ts: Optional[int] = None,      # unix seconds
        # Core defaults
        default_reputation: float = 0.5,
        conservatism: float = 0.8,
        # Weighting preferences
        scale: int = 10**8,
        weight_mode: WeightMode = "raw",
        use_liquid_rater_weight: bool = True,
        # Numerics
        epsilon: float = 1e-12,
    ):
        self.period = period
        self.start_ts = start_ts
        self.end_ts = end_ts

        self.default_reputation = default_reputation
        self.conservatism = conservatism

        self.scale = scale
        self.weight_mode = weight_mode
        self.use_liquid_rater_weight = use_liquid_rater_weight

        self.epsilon = epsilon



# ---------------------------------------------------------------------
# Data initializer
# ---------------------------------------------------------------------
class LiquidRankDataInitializer:
    """
    Loads raw events, normalizes timestamps, sorts, optionally applies cutoff,
    optionally filters by cfg.start_ts/end_ts, and outputs a transactions list.

    Produces transactions compatible with your existing code:
      {'from': ..., 'to': ..., 'value': ..., 'time': ...}
    """

    def __init__(
        self,
        cfg: LiquidRankConfig,
        *,
        timestamp_col: str = "timestamp",
        from_col: str = "from",
        to_col: str = "to",
        value_col: str = "value",
    ):
        self.cfg = cfg
        self.timestamp_col = timestamp_col
        self.from_col = from_col
        self.to_col = to_col
        self.value_col = value_col

    def _ensure_required_columns(self, df: pd.DataFrame) -> None:
        required = [self.from_col, self.to_col, self.value_col, self.timestamp_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def load_csv(self, path: str) -> pd.DataFrame:
        logger.info("Loading CSV: %s", path)
        df = pd.read_csv(path)
        return df

    def parse_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp to datetime (UTC) and sort ascending.
        """
        self._ensure_required_columns(df)

        out = df.copy()
        out[self.timestamp_col] = pd.to_datetime(out[self.timestamp_col], utc=True, errors="coerce")
        if out[self.timestamp_col].isna().any():
            bad = df.loc[out[self.timestamp_col].isna(), self.timestamp_col].head(5).tolist()
            raise ValueError(f"Could not parse some timestamps in '{self.timestamp_col}'. Examples: {bad}")

        out = out.sort_values(self.timestamp_col).reset_index(drop=True)
        return out

    def apply_cutoff_date(
        self,
        df: pd.DataFrame,
        cutoff_date: Optional[Union[datetime, str]] = None
    ) -> pd.DataFrame:
        """
        Optional: filter to simulate up to a certain date (inclusive).
        """
        if cutoff_date is None:
            return df

        cutoff_dt = pd.to_datetime(cutoff_date, utc=True)
        out = df[df[self.timestamp_col] <= cutoff_dt].copy()

        logger.info("Applied cutoff <= %s. Rows: %d -> %d", cutoff_dt, len(df), len(out))
        return out

    def apply_time_window_from_config(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional: filter using cfg.start_ts and cfg.end_ts (unix seconds).
        """
        out = df

        if self.cfg.start_ts is not None:
            start_dt = pd.to_datetime(self.cfg.start_ts, unit="s", utc=True)
            out = out[out[self.timestamp_col] >= start_dt]

        if self.cfg.end_ts is not None:
            end_dt = pd.to_datetime(self.cfg.end_ts, unit="s", utc=True)
            out = out[out[self.timestamp_col] <= end_dt]

        if len(out) != len(df):
            logger.info("Applied config time window. Rows: %d -> %d", len(df), len(out))

        return out.copy()

    def to_transactions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Builds a list of transactions, bucketing time to cfg.period.
        For cfg.period='D', this is equivalent to your `.normalize()` (daily date).
        """
        times = df[self.timestamp_col].dt.floor(self.cfg.period)

        txs: List[Dict[str, Any]] = []
        for i, row in df.iterrows():
            txs.append({
                "from": row[self.from_col],
                "to": row[self.to_col],
                "value": row[self.value_col],
                "time": times.loc[i],
            })

        logger.info("Prepared %d transactions using period=%s.", len(txs), self.cfg.period)
        return txs

    def initialize_from_csv(
        self,
        path: str,
        *,
        cutoff_date: Optional[Union[datetime, str]] = None
    ) -> Dict[str, Any]:
        """
        One-call initialization:
          load -> parse/sort -> cutoff -> cfg window -> transactions
        """
        df = self.load_csv(path)
        df = self.parse_and_sort(df)
        df = self.apply_cutoff_date(df, cutoff_date=cutoff_date)
        df = self.apply_time_window_from_config(df)
        df = self.enrich_initialized_frame(df)
        transactions = self.to_transactions(df)

        return {"df": df, "transactions": transactions}

    # -----------------------------------------------------------------
    # Next initialization steps: normalize_value, weight, tx metadata
    # -----------------------------------------------------------------

    def ensure_optional_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure optional columns exist so downstream logic can rely on them.
        This does not overwrite existing values.
        """
        out = df.copy()

        if "transactionID" not in out.columns:
            # Deterministic fallback ID (stable given the sort)
            out["transactionID"] = np.arange(len(out), dtype=np.int64)

        if "transaction_type" not in out.columns:
            out["transaction_type"] = "transfer"

        return out

    def add_normalized_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures 'normalized_value' exists and is in [0, 1].
        Default implementation: min-max normalize 'value'.
        """
        out = df.copy()

        if "normalized_value" in out.columns:
            # Validate/clean a bit (do not hard-fail unless NaNs)
            nv = pd.to_numeric(out["normalized_value"], errors="coerce")
            if nv.isna().any():
                bad = out.loc[nv.isna(), "normalized_value"].head(5).tolist()
                raise ValueError(f"'normalized_value' contains non-numeric values. Examples: {bad}")
            ## out["normalized_value"] = nv.clip(lower=0.0, upper=1.0)
            return raise ValueError(f"'Normalized values are above 1 or below 0.")

        v = pd.to_numeric(out[self.value_col], errors="coerce")
        if v.isna().any():
            bad = out.loc[v.isna(), self.value_col].head(5).tolist()
            raise ValueError(f"'{self.value_col}' contains non-numeric values. Examples: {bad}")

        vmin = float(v.min())
        vmax = float(v.max())
        denom = vmax - vmin

        if denom <= self.cfg.epsilon:
            # All values identical (or effectively so) => uniform normalized_value
            out["normalized_value"] = 1.0
        else:
            out["normalized_value"] = ((v - vmin) / denom).astype(float)

        return out

    def add_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures 'weight' exists.
        Default: weight = normalized_value * scale.
        """
        out = df.copy()

        if "weight" in out.columns:
            w = pd.to_numeric(out["weight"], errors="coerce")
            if w.isna().any():
                bad = out.loc[w.isna(), "weight"].head(5).tolist()
                raise ValueError(f"'weight' contains non-numeric values. Examples: {bad}")
            out["weight"] = w.astype(float)
            return out

        if "normalized_value" not in out.columns:
            raise KeyError("Missing 'normalized_value'. Call add_normalized_value() first.")

        out["weight"] = out["normalized_value"].astype(float) * float(self.cfg.scale)
        return out

    def enrich_initialized_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the next initialization steps after parse/sort and time filtering:
          - ensure optional metadata columns
          - add normalized_value (if missing)
          - add weight (if missing)
          - apply weight_mode transform
        """
        out = self.ensure_optional_columns(df)
        out = self.add_normalized_value(out)
        out = self.add_weight(out)
        return out



        

class ReputationService(abc.ABC):
    """
    Concrete parameter store for Liquid Rank / reputation simulation.

    This class *only* manages parameters and defaults for now.
    """

    # Error codes
    ERR_NOT_A_DICT = 1
    ERR_UNKNOWN_KEY = 2
    ERR_INVALID_TYPE = 3
    ERR_INVALID_VALUE = 4

    def __init__(self) -> None:
        # Canonical parameter set with defaults
        self._params: Dict[str, Any] = {
            # Core
            "default_reputation": 0.5,
            "conservatism": 0.1,     # matches your example (conservativity=0.1)
            "precision": 1,

            # reputation_calc_p1 flags
            "temporal_aggregation": False,
            "need_occurance": False,
            "logratings": False,
            "downrating": False,
            "weighting": True,
            "rater_bias": None,
            "averages": None,

            # calculate_new_reputation flags
            "rating": True,
            "unrated": False,
            "normalizedRanks": True,
            "denomination": True,
            "liquid": False,
            "logranks": True,

            # predictive settings
            "predictiveness": 0,
            "predictive_data": {},
        }

        # Define allowed keys once (prevents silent typos)
        self._allowed_keys = set(self._params.keys())

    def set_parameters(self, parameters: Dict[str, Any]) -> int:
        """
        Partial update: only keys present in `parameters` are updated.
        Unknown keys are rejected with ERR_UNKNOWN_KEY.
        """
        if not isinstance(parameters, dict):
            return self.ERR_NOT_A_DICT

        # Reject unknown keys first (fail fast)
        for k in parameters.keys():
            if k not in self._allowed_keys:
                return self.ERR_UNKNOWN_KEY

        # Validate and apply
        for k, v in parameters.items():
            code = self._validate_one(k, v)
            if code != 0:
                return code
            self._params[k] = v

        return 0

    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns a defensive copy so callers cannot mutate internal state.
        """
        return deepcopy(self._params)

    # -----------------------
    # Validation
    # -----------------------
    def _validate_one(self, key: str, value: Any) -> int:
        # Core numeric validations
        if key == "default_reputation":
            if not isinstance(value, (int, float)):
                return self.ERR_INVALID_TYPE
            if not (0.0 <= float(value) <= 1.0):
                return self.ERR_INVALID_VALUE
            return 0

        if key == "conservatism":
            if not isinstance(value, (int, float)):
                return self.ERR_INVALID_TYPE
            # allow 0..1 inclusive
            if not (0.0 <= float(value) <= 1.0):
                return self.ERR_INVALID_VALUE
            return 0

        if key == "precision":
            if not isinstance(value, int):
                return self.ERR_INVALID_TYPE
            if value < 0:
                return self.ERR_INVALID_VALUE
            return 0

        # Booleans
        bool_keys = {
            "temporal_aggregation", "need_occurance", "logratings", "downrating",
            "weighting", "rating", "unrated", "normalizedRanks", "denomination",
            "liquid", "logranks",
        }
        if key in bool_keys:
            if not isinstance(value, bool):
                return self.ERR_INVALID_TYPE
            return 0

        # Predictive settings
        if key == "predictiveness":
            if not isinstance(value, int):
                return self.ERR_INVALID_TYPE
            if value < 0:
                return self.ERR_INVALID_VALUE
            return 0

        if key == "predictive_data":
            if not isinstance(value, dict):
                return self.ERR_INVALID_TYPE
            return 0

        # Optional passthroughs
        if key in {"rater_bias", "averages"}:
            # allow any object or None for now
            return 0

        # Should not happen because unknown keys rejected earlier
        return self.ERR_UNKNOWN_KEY


"""
Reputation Rating Service interface definition
"""        
class RatingService(ReputationService):

    """
    In-memory rating store.

    Stores ratings as a list of normalized dicts with keys:
      from, type, to, value, weight, time

    Time is stored as pandas.Timestamp (UTC if timezone-naive inputs are provided).
    """

    # Error codes
    ERR_NOT_A_DICT = 1
    ERR_NOT_A_LIST = 2
    ERR_INVALID_TYPE = 3
    ERR_INVALID_VALUE = 4

    def __init__(self) -> None:
        # Parameters (basic baseline; you can extend later)
        self._params: Dict[str, Any] = {
            "default_reputation": 0.5,
            "conservatism": 0.1,
            "precision": 1,
        }
        self._allowed_keys = set(self._params.keys())

        # Ratings storage
        self._ratings: List[Dict[str, Any]] = []

    # -------------------------
    # ReputationService methods
    # -------------------------
    def set_parameters(self, parameters: Dict[str, Any]) -> int:
        if not isinstance(parameters, dict):
            return self.ERR_NOT_A_DICT

        for k in parameters.keys():
            if k not in self._allowed_keys:
                return self.ERR_INVALID_VALUE

        # Minimal validation (can be expanded)
        for k, v in parameters.items():
            if k in {"default_reputation", "conservatism"}:
                if not isinstance(v, (int, float)):
                    return self.ERR_INVALID_TYPE
            if k == "precision":
                if not isinstance(v, int) or v < 0:
                    return self.ERR_INVALID_VALUE

            self._params[k] = v

        return 0

    def get_parameters(self) -> Dict[str, Any]:
        return deepcopy(self._params)

    # -------------------------
    # RatingService methods
    # -------------------------
    def put_ratings(self, ratings: List[Dict[str, Any]]) -> int:
        """
        ratings: list of dicts with keys:
          from (str), type (str), to (str), value (number), weight (number), time (datetime-like)
        """
        if not isinstance(ratings, list):
            return self.ERR_NOT_A_LIST

        normalized: List[Dict[str, Any]] = []

        for item in ratings:
            if not isinstance(item, dict):
                return self.ERR_INVALID_TYPE

            # Required keys
            required = {"from", "type", "to", "value", "weight", "time"}
            if not required.issubset(item.keys()):
                return self.ERR_INVALID_VALUE

            frm = item["from"]
            typ = item["type"]
            to = item["to"]
            val = item["value"]
            wgt = item["weight"]
            t = item["time"]

            if not isinstance(frm, str) or not isinstance(to, str) or not isinstance(typ, str):
                return self.ERR_INVALID_TYPE
            if not isinstance(val, (int, float)) or not isinstance(wgt, (int, float)):
                return self.ERR_INVALID_TYPE

            # Normalize time to pandas Timestamp (UTC)
            ts = pd.to_datetime(t, utc=True, errors="coerce")
            if pd.isna(ts):
                return self.ERR_INVALID_VALUE

            normalized.append({
                "from": frm,
                "type": typ,
                "to": to,
                "value": float(val),
                "weight": float(wgt),
                "time": ts,
            })

        # Append (no dedupe yet; we can add transactionID later if needed)
        self._ratings.extend(normalized)
        return 0

    def get_ratings(self, filter: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
        """
        filter:
          since: datetime-like (inclusive)
          until: datetime-like (inclusive)
          ids: list[str] (incoming OR outgoing)
          from: list[str] (outgoing only)
          to: list[str] (incoming only)
        """
        if not isinstance(filter, dict):
            return self.ERR_NOT_A_DICT, []

        # Parse times (optional)
        since = filter.get("since", None)
        until = filter.get("until", None)

        since_ts = pd.to_datetime(since, utc=True, errors="coerce") if since is not None else None
        until_ts = pd.to_datetime(until, utc=True, errors="coerce") if until is not None else None

        if since is not None and pd.isna(since_ts):
            return self.ERR_INVALID_VALUE, []
        if until is not None and pd.isna(until_ts):
            return self.ERR_INVALID_VALUE, []

        # Parse ID filters (optional)
        ids = filter.get("ids", None)
        from_ids = filter.get("from", None)
        to_ids = filter.get("to", None)

        if ids is not None and (not isinstance(ids, list) or any(not isinstance(x, str) for x in ids)):
            return self.ERR_INVALID_TYPE, []
        if from_ids is not None and (not isinstance(from_ids, list) or any(not isinstance(x, str) for x in from_ids)):
            return self.ERR_INVALID_TYPE, []
        if to_ids is not None and (not isinstance(to_ids, list) or any(not isinstance(x, str) for x in to_ids)):
            return self.ERR_INVALID_TYPE, []

        # Filter ratings
        out: List[Dict[str, Any]] = []
        for r in self._ratings:
            t = r["time"]

            if since_ts is not None and t < since_ts:
                continue
            if until_ts is not None and t > until_ts:
                continue

            # ids => incoming or outgoing
            if ids is not None:
                if (r["from"] not in ids) and (r["to"] not in ids):
                    continue

            # from => outgoing only
            if from_ids is not None:
                if r["from"] not in from_ids:
                    continue

            # to => incoming only
            if to_ids is not None:
                if r["to"] not in to_ids:
                    continue

            out.append(r)

        # Return copies with time serialized to ISO if you prefer; for now keep Timestamp
        # (keeping Timestamp is more useful for downstream bucketing)
        return 0, [dict(x) for x in out]

    def clear_ratings(self) -> int:
        self._ratings.clear()
        return 0

"""
Reputation Ranking Service interface definition
"""        
class RankingService(ReputationService):

    """
    A minimal, deterministic storage-backed ranking service.

    Stores ranks by date:
      self._ranks_by_date[date_key] = {id: rank, ...}

    This is intentionally decoupled from the computation logic:
    - update_ranks(date) will be wired to the algorithm later
    - put/get/clear are ready now
    """

    # Error codes
    ERR_NOT_A_DICT = 1
    ERR_UNKNOWN_KEY = 2
    ERR_INVALID_TYPE = 3
    ERR_INVALID_VALUE = 4
    ERR_NOT_IMPLEMENTED = 5
    ERR_NOT_FOUND = 6

    def __init__(self) -> None:
        # Parameter store (same idea as earlier); defaults can be extended later
        self._params: Dict[str, Any] = {
            "default_reputation": 0.5,
            "conservatism": 0.1,
            "precision": 1,
        }
        self._allowed_keys = set(self._params.keys())

        # Rank storage
        self._ranks_by_date: Dict[date_type, Dict[str, float]] = {}

    # -------------------------
    # ReputationService methods
    # -------------------------
    def set_parameters(self, parameters: Dict[str, Any]) -> int:
        if not isinstance(parameters, dict):
            return self.ERR_NOT_A_DICT

        for k in parameters.keys():
            if k not in self._allowed_keys:
                return self.ERR_UNKNOWN_KEY

        for k, v in parameters.items():
            code = self._validate_param(k, v)
            if code != 0:
                return code
            self._params[k] = v

        return 0

    def get_parameters(self) -> Dict[str, Any]:
        return deepcopy(self._params)

    # -------------------------
    # RankingService methods
    # -------------------------
    def update_ranks(self, date) -> int:
        """
        Placeholder until computation is implemented.
        Later this will:
          - run the liquid-rank update for the period ending at `date`
          - store results into self._ranks_by_date[date]
        """
        _ = self._coerce_date(date)  # validate date
        return self.ERR_NOT_IMPLEMENTED

    def put_ranks(self, date, ranks) -> int:
        """
        Store ranks for a given date.
        ranks must be: [{"id": <str>, "rank": <number>}, ...]
        """
        d = self._coerce_date(date)

        if not isinstance(ranks, list):
            return self.ERR_INVALID_TYPE

        bucket: Dict[str, float] = {}
        for item in ranks:
            if not isinstance(item, dict):
                return self.ERR_INVALID_TYPE
            if "id" not in item or "rank" not in item:
                return self.ERR_INVALID_VALUE

            rid = item["id"]
            rnk = item["rank"]

            if not isinstance(rid, str):
                return self.ERR_INVALID_TYPE
            if not isinstance(rnk, (int, float)):
                return self.ERR_INVALID_TYPE

            bucket[rid] = float(rnk)

        self._ranks_by_date[d] = bucket
        return 0

    def get_ranks(self, filter) -> Tuple[int, List[Dict[str, Any]]]:
        """
        filter:
          - date (required)
          - ids (optional list[str])
        """
        if not isinstance(filter, dict):
            return self.ERR_NOT_A_DICT, []

        if "date" not in filter:
            return self.ERR_INVALID_VALUE, []

        d = self._coerce_date(filter["date"])

        if d not in self._ranks_by_date:
            return self.ERR_NOT_FOUND, []

        ranks_dict = self._ranks_by_date[d]

        ids = filter.get("ids", None)
        if ids is None:
            # Return all ranks
            out = [{"id": k, "rank": v} for k, v in ranks_dict.items()]
            return 0, out

        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            return self.ERR_INVALID_TYPE, []

        out = [{"id": i, "rank": ranks_dict[i]} for i in ids if i in ranks_dict]
        return 0, out

    def clear_ranks(self) -> int:
        ### Clears ranks. Best to avoid this unless ending one simulation and starting another.
        self._ranks_by_date.clear()
        return 0

    def get_ranks_dict(self, filter) -> Tuple[int, Dict[str, float]]:
        """
        Same filter as get_ranks, but returns a dict {id: rank}.
        """
        code, rows = self.get_ranks(filter)
        if code != 0:
            return code, {}
        return 0, {row["id"]: float(row["rank"]) for row in rows}

    # -------------------------
    # Helpers
    # -------------------------
    def _coerce_date(self, d: Union[date_type, datetime, str]) -> date_type:
        """
        Accepts:
          - datetime.date
          - datetime.datetime
          - ISO date string (YYYY-MM-DD) or datetime string parseable by pandas
        Returns: datetime.date
        """
        if isinstance(d, date_type) and not isinstance(d, datetime):
            return d
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, str):
            # robust parsing
            parsed = pd.to_datetime(d, utc=True, errors="raise")
            return parsed.date()
        raise TypeError("date must be datetime.date, datetime.datetime, or parseable string")

    def _validate_param(self, key: str, value: Any) -> int:
        if key == "default_reputation":
            if not isinstance(value, (int, float)):
                return self.ERR_INVALID_TYPE
            if not (0.0 <= float(value) <= 1.0):
                return self.ERR_INVALID_VALUE
            return 0

        if key == "conservatism":
            if not isinstance(value, (int, float)):
                return self.ERR_INVALID_TYPE
            if not (0.0 <= float(value) <= 1.0):
                return self.ERR_INVALID_VALUE
            return 0

        if key == "precision":
            if not isinstance(value, int):
                return self.ERR_INVALID_TYPE
            if value < 0:
                return self.ERR_INVALID_VALUE
            return 0

        return self.ERR_UNKNOWN_KEY
