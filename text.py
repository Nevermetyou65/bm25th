"""BM25 Transformer"""

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.fixes import _IS_32BIT
from sklearn.preprocessing import normalize
from sklearn.utils._param_validation import StrOptions, RealNotInt


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.issparse(X) and X.format == "csr":
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


class BM25Transformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a BM25 representation.

    BM25 is a ranking function used in information retrieval. Similar to TF-IDF
    but with better handling of term frequency saturation and document length normalization.

    Parameters
    ----------
    norm : {'l1', 'l2'} or None, default='l2'
        Normalization to apply after BM25 scoring.

    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting.

    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies.

    sublinear_tf : bool, default=False
        Apply sublinear tf scaling (1 + log(tf)).

    k1 : float, default=1.5
        Controls term frequency saturation. Higher values increase the impact of term frequency.

    b : float, default=0.75
        Controls document length normalization. 0 = no normalization, 1 = full normalization.
    """

    _parameter_constraints: dict = {
        "norm": [StrOptions({"l1", "l2"}), None],
        "use_idf": ["boolean"],
        "smooth_idf": ["boolean"],
        "sublinear_tf": ["boolean"],
        "k1": [RealNotInt],
        "b": [RealNotInt],
    }

    def __init__(
        self,
        *,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        k1=1.5,
        b=0.75,
    ):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.k1 = k1
        self.b = b

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Learn the IDF vector (global term weights).

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            A matrix of term/token counts.

        y : None
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = validate_data(
            self, X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in (np.float64, np.float32) else np.float64

        if self.use_idf:
            n_samples, _ = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, copy=False)
            # df is the n(qi) in https://en.wikipedia.org/wiki/Okapi_BM25

            # perform idf smoothing if required
            df += float(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            self.idf_ = (
                (np.full_like(df, fill_value=n_samples, dtype=dtype) - df + 0.5)
                / (df + 0.5)
            ) + 1
            np.log(self.idf_, out=self.idf_)

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to BM25 representation.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            A matrix of term/token counts.

        copy : bool, default=True
            Whether to copy X before transforming.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            BM25-weighted document-term matrix.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            copy=copy,
            reset=False,
        )
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=X.dtype)

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1.0

        if hasattr(self, "idf_"):
            # Apply BM25 term frequency scaling
            doc_lens = np.array(X.sum(axis=1)).flatten()
            avgdl = doc_lens.mean()

            # Map each entry to its document
            doc_indices = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
            doc_len_per_entry = doc_lens[doc_indices]

            # BM25 formula: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            X.data = (X.data * (self.k1 + 1)) / (
                X.data + self.k1 * (1 - self.b + self.b * (doc_len_per_entry / avgdl))
            )

            # Apply IDF weights
            X.data *= self.idf_[X.indices]

        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)

        return X
