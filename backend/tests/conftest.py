import os

# the test suite exercises the deterministic code paths; the local language model is tested end to end separately
os.environ.setdefault("CWE_DISABLE_LOCAL_AI", "1")
