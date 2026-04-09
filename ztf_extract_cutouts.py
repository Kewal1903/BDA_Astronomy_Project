import os, io, gzip, glob
from astropy.io import fits
from fastavro import reader

IN_DIR  = "/data/ztf_dl"
OUT_DIR = "/data/ztf_fits"
os.makedirs(OUT_DIR, exist_ok=True)

def write_cutout(bytes_gz, out_path):
    if not bytes_gz:
        return
    with gzip.GzipFile(fileobj=io.BytesIO(bytes_gz)) as gz:
        raw = gz.read()
    with fits.open(io.BytesIO(raw), memmap=False) as hdul:
        hdul.writeto(out_path, overwrite=True)

count = 0
for avro_path in glob.glob(os.path.join(IN_DIR, "*.avro")):
    with open(avro_path, "rb") as f:
        for rec in reader(f):
            cid = rec.get("candid") or rec.get("alertId") or count
            for kind in ("Science","Template","Difference"):
                data = (rec.get(f"cutout{kind}") or {}).get("stampData")
                if data:
                    out = os.path.join(OUT_DIR, f"{cid}_{kind.lower()}.fits")
                    write_cutout(data, out)
                    count += 1
print("Wrote FITS cutouts:", count, "to", OUT_DIR)
