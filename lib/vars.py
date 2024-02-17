from pathlib import Path


DB_NAME = "BI"
DB_IP = "10.171.18.144"


## Change as needed
DS_PREFIX = Path("S:/").expanduser()

## List of Available datasets
AVAIL_DS = {_.name for _ in DS_PREFIX.iterdir() if _.is_dir()}



# DB_CON = f"mongodb://{DB_IP}/{DB_NAME}"