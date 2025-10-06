from typing import List

# Minimal fallback list. If `airportsdata` is available, use it to populate
# real IATA codes; otherwise expose a small static list to satisfy imports.
AIRPORT_LIST: List[str] = []
try:
    import airportsdata  # type: ignore
    try:
        data = airportsdata.load("IATA")
    except Exception:
        data = airportsdata.load()
    if isinstance(data, dict):
        AIRPORT_LIST = sorted(list(data.keys()))
except Exception:
    AIRPORT_LIST = ["KUL", "SIN", "JFK", "LHR"]

