import re
import pandas as pd
import matplotlib.pyplot as plt

# Load & clean raw lines
LOG_FILE = "raw_data/BGL_2k.log"

lines = []
with open(LOG_FILE) as f:
    for line in f:
        raw = line.strip()
        # strip leading hyphens/spaces
        raw = re.sub(r'^[\-\s]+', '', raw)
        if raw:
            lines.append(raw)
print(f"Read {len(lines)} non-empty lines from {LOG_FILE}")

# Parse each line
pattern = re.compile(
    r'^(?P<epoch>\d+)\s+'              
    r'(?P<date>\S+)\s+'               
    r'(?P<node>\S+)\s+'               
    r'(?P<ts2>\S+)\s+'                
    r'(?P=node)\s+'                   
    r'RAS\s+(?P<component>\w+)\s+'    
    r'(?P<severity>\w+)\s+'           
    r'(?P<message>.+)$'               
)

records = []
for L in lines:
    m = pattern.match(L)
    if not m:
        continue
    d = m.groupdict()
    d['dt'] = pd.to_datetime(d['ts2'], format='%Y-%m-%d-%H.%M.%S.%f')
    records.append(d)

df = pd.DataFrame(records)
print(f"Parsed {len(df)} log entries.")
