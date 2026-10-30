"""ACT-TSFM research implementation."""
import os
# Small posterior matrices are faster without BLAS thread oversubscription.
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
