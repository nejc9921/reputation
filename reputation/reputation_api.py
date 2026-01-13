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
        transactions = self.to_transactions(df)

        return {"df": df, "transactions": transactions}
        
"""
Reputation Generic Service interface definition
"""        
class ReputationService(abc.ABC):

	"""
	Input: dict of all parameters that needs to be set (not listed parameters are not affected)
	Output: 0 on success, integer error code on error 
	"""
	@abc.abstractmethod
	def set_parameters(self,parameters):
		pass

	@abc.abstractmethod
	def get_parameters(self):
		pass


"""
Reputation Rating Service interface definition
"""        
class RatingService(ReputationService):

	"""
	Input: List of dicts with the key-value pairs for the attributes: "from","type","to","value","weight","time"
	Output: 0 on success, integer error code on error 
	"""
	@abc.abstractmethod
	def put_ratings(self,ratings):
		pass

	"""
	Input: filter as dict of the following:
		since - starting time inclusively
		until - ending time inclusively
		ids - list of ids to retrieve incoming AND outgoing ratings BOTH
		from - list of ids to retrieve outgoing ratings ONLY (TODO later)
		to - list of ids to retrieve incoming ratings ONLY (TODO later)
	Output: tuple of the pair:
		0 on success, integer error code on error
		List of dicts with the key-value pairs for the attributes: "from","type","to","value","weight","time"
	"""
	@abc.abstractmethod
	def get_ratings(self,filter):
		pass

	"""
	Input: None
	Output: 0 on success, integer error code on error 
	"""
	@abc.abstractmethod
	def clear_ratings(self):
		pass
		
"""
Reputation Ranking Service interface definition
"""        
class RankingService(ReputationService):

	"""
	Input: Date to update the ranks for
	Output: 0 on success, integer error code on error 
	"""
	@abc.abstractmethod
	def update_ranks(self,date):
		pass

	"""
	Input: Date and list of dicts with two key-value pairs for "id" and "rank" 
	Output: 0 on success, integer error code on error 
	"""
	@abc.abstractmethod
	def put_ranks(self,date,ranks):
		pass

	"""
	Input: filter as dict of the following:
		date - date to provide the ranks
		ids - list of ids to retrieve the ranks
	Output: tuple of the pair:
		0 on success, integer error code on error
		List of dicts with the two key-value pairs for "id" and "rank"
	"""
	@abc.abstractmethod
	def get_ranks(self,filter):
		pass

	"""
	Input: None
	Output: 0 on success, integer error code on error 
	"""
	@abc.abstractmethod
	def clear_ranks(self):
		pass

	@abc.abstractmethod
	def get_ranks_dict(self,filter):
		pass
