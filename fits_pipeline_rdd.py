# app/fits_pipeline_rdd.py
import io
import numpy as np
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("FITS-RDD-Pipeline")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
    .config("spark.executor.memory", "1g")
    .config("spark.executor.cores", "1")
    .getOrCreate()
)
sc = spark.sparkContext

INPUT = "hdfs://namenode:8020/user/astro/raw/ztf_fits/*.fits"
fits_rdd = sc.binaryFiles(INPUT, minPartitions=4)  # (path, bytes) - reduced partitions

def process_partition(records):
    import io, warnings
    import numpy as np
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyWarning
    warnings.simplefilter("ignore", VerifyWarning)

    out = []
    for path, data in records:
        try:
            with fits.open(io.BytesIO(data), memmap=False) as hdul:
                hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                if hdu is None or hdu.data is None:
                    out.append((path, None, None, "ERR:NoImageData"))
                    continue
                img = np.asarray(hdu.data, dtype=np.float32)
                img = np.where(np.isfinite(img), img, np.nan)
                if np.isnan(img).all():
                    out.append((path, None, None, "ERR:AllNaN"))
                    continue
                med = float(np.nanmedian(img))
                img = img - med
                mean = float(np.nanmean(img))
                std  = float(np.nanstd(img))
                thr  = mean + 3.0 * std if np.isfinite(std) else np.nan
                bright = int(np.nansum(img > thr)) if np.isfinite(thr) else 0
                out.append((path, mean, std, bright))
        except Exception as e:
            out.append((path, None, None, f"ERR:{e}"))
    return iter(out)

# Build once

# build stats_rdd
stats_rdd = fits_rdd.mapPartitions(process_partition)

# don't squash to 8; either keep Spark's many partitions or make it moderately large

stats_rdd = stats_rdd.coalesce(4)         # ✅ reduced for limited resources

# cache so retries don't re-run FITS parsing
from pyspark import StorageLevel
stats_rdd = stats_rdd.persist(StorageLevel.MEMORY_AND_DISK)

from datetime import datetime
run_tag = datetime.utcnow().strftime("full_%Y%m%d_%H%M%S")
tmp   = f"hdfs://namenode:8020/user/spark/bronze/fits_stats_{run_tag}.__tmp__"
final = f"hdfs://namenode:8020/user/spark/bronze/fits_stats_{run_tag}"


#stats_rdd.map(lambda t: ",".join(map(str, t))).saveAsTextFile(tmp)

# atomic rename (unchanged)
# ensure output locations are clean before first write
jvm  = spark._jvm
conf = spark._jsc.hadoopConfiguration()
fs   = jvm.org.apache.hadoop.fs.FileSystem.get(conf)
Path = jvm.org.apache.hadoop.fs.Path

tmpP, finalP = Path(tmp), Path(final)
if fs.exists(tmpP):
    fs.delete(tmpP, True)
if fs.exists(finalP):
    fs.delete(finalP, True)


stats_rdd.map(lambda t: ",".join(map(str, t))).saveAsTextFile(tmp)

ok = fs.rename(tmpP, finalP)
print(f"rename {tmp} -> {final} : {ok}")
if not ok:
    raise RuntimeError("Failed to rename tmp output to final location")


spark.stop()

