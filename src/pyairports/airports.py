from typing import List

# Minimal compatibility structure: outlines expects AIRPORT_LIST to be an
# iterable of records where record[3] is the IATA code (non-empty).
# We provide lightweight records of the form ["", "", "", IATA_CODE].

AIRPORT_LIST: List[List[str]] = []

try:
    import airportsdata  # type: ignore

    try:
        data = airportsdata.load("IATA")  # dict: IATA -> {...}
    except Exception:
        data = airportsdata.load()

    if isinstance(data, dict):
        codes = sorted(k for k in data.keys() if isinstance(k, str) and k)
        AIRPORT_LIST = [["", "", "", code] for code in codes]
except Exception:
    # Minimal fallback set of common IATA codes
    fallback = ["KUL", "SIN", "JFK", "LHR", "HND", "DXB"]
    AIRPORT_LIST = [["", "", "", code] for code in fallback]
