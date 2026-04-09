import io
import warnings
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

def extract_fits_features(records):
    """
    This function is shipped to all CPU cores. It opens the raw FITS binary,
    extracts the metadata from the filename, and calculates the image math.
    """
    warnings.simplefilter("ignore", VerifyWarning)

    results = []
    for path, data in records:
        try:
            # 1. Extract metadata from the file path
            filename = path.split("/")[-1]
            parts = filename.replace(".fits", "").split("_")
            object_id = parts[0]
            img_type = parts[1].upper() if len(parts) > 1 else "UNKNOWN"

            # 2. Open the image file in memory
            with fits.open(io.BytesIO(data), memmap=False) as hdul:
                hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                if hdu is None or hdu.data is None:
                    results.append((path, object_id, img_type, None, None, None, None, "ERR:NoImageData"))
                    continue

                img = np.asarray(hdu.data, dtype=np.float32)
                img = np.where(np.isfinite(img), img, np.nan)

                if np.isnan(img).all():
                    results.append((path, object_id, img_type, None, None, None, None, "ERR:AllNaN"))
                    continue

                # 3. The Math (Background noise and brightness thresholding)
                med = float(np.nanmedian(img))
                img_sub = img - med
                mean = float(np.nanmean(img_sub))
                std  = float(np.nanstd(img_sub))
                thr  = mean + 3.0 * std if np.isfinite(std) else np.nan
                
                # Count how many pixels are "bright" (potential objects)
                bright_pixels = int(np.nansum(img_sub > thr)) if np.isfinite(thr) else 0

                results.append((path, object_id, img_type, med, mean, std, bright_pixels, "SUCCESS"))
                
        except Exception as e:
            results.append((path, object_id, "UNKNOWN", None, None, None, None, f"ERR:{str(e)[:20]}"))

    return iter(results)


if __name__ == "__main__":
    # Initialize Spark for YARN Cluster processing
    spark = SparkSession.builder \
        .appName("01_Bronze_Pipeline_Cluster") \
        .getOrCreate()

    # SILENCE THE NOISE: Only show Warnings and Errors in the terminal
    spark.sparkContext.setLogLevel("WARN")

    # Define HDFS paths for the 340-file subset
    INPUT_DIR = "hdfs://master:9000/user/ztf/subset_12k/*.fits"
    OUTPUT_DIR = "hdfs://master:9000/user/ztf/subset_12k_bronze.parquet"

    # 1. Read the binary files from HDFS
    rdd = spark.sparkContext.binaryFiles(INPUT_DIR)

    # 2. Process them in parallel using the worker nodes
    processed_rdd = rdd.mapPartitions(extract_fits_features)

    # 3. Map the results to a strict Schema
    schema = StructType([
        StructField("file_path", StringType(), True),
        StructField("object_id", StringType(), True),
        StructField("image_type", StringType(), True),
        StructField("median_bg", FloatType(), True),
        StructField("mean_sub", FloatType(), True),
        StructField("std_sub", FloatType(), True),
        StructField("bright_pixels", IntegerType(), True),
        StructField("status", StringType(), True)
    ])

    df = spark.createDataFrame(processed_rdd, schema)

    # 4. Trigger the job and save it as a highly compressed Parquet file on HDFS
    print("\n[SYSTEM] Submitting FITS extraction to YARN worker nodes...")
    df.write.mode("overwrite").parquet(OUTPUT_DIR)

    # 5. Display a sample in the terminal to verify success
    print("\n[SUCCESS] Bronze Layer Parquet generated. Here is a sample:")
    df.show(10)
    print(f"\n[INFO] Data safely written to: {OUTPUT_DIR}\n")
    
    spark.stop()
