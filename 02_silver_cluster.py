from pyspark.sql import SparkSession
from pyspark.sql.functions import col, first

if __name__ == "__main__":
    # Initialize Spark for YARN Cluster processing
    spark = SparkSession.builder \
        .appName("02_Silver_Pipeline_Cluster") \
        .getOrCreate()

    # SILENCE THE NOISE: Only show Warnings and Errors
    spark.sparkContext.setLogLevel("WARN")

    # Define HDFS paths linking Bronze output to Silver input
    INPUT_DIR = "hdfs://master:9000/user/ztf/subset_12k_bronze.parquet"
    OUTPUT_DIR = "hdfs://master:9000/user/ztf/subset_12k_silver.parquet"

    print(f"\n[SYSTEM] Reading Bronze Data from: {INPUT_DIR}")
    # 1. Read the Bronze Parquet data
    df_bronze = spark.read.parquet(INPUT_DIR)

    # 2. Clean: Keep only successful extractions and drop unnecessary columns
    df_clean = df_bronze.filter(col("status") == "SUCCESS").drop("status", "file_path")

    # 3. Pivot: Collapse 3 rows into 1 row per object_id
    print("[SYSTEM] Pivoting data structure to create wide ML features...")
    df_silver = df_clean.groupBy("object_id").pivot("image_type").agg(
        first("median_bg").alias("median_bg"),
        first("mean_sub").alias("mean_sub"),
        first("std_sub").alias("std_sub"),
        first("bright_pixels").alias("bright_pixels")
    )

    # 4. Drop any objects that are missing one of the 3 images (Science, Template, or Difference)
    df_silver = df_silver.na.drop()

    # 5. Trigger the execution and save the new Parquet file
    print("[SYSTEM] Submitting Silver transformation to YARN worker nodes...")
    df_silver.write.mode("overwrite").parquet(OUTPUT_DIR)

    # 6. Display the beautifully structured ML-ready data
    print("\n[SUCCESS] Silver Layer complete! Data safely written. Here is a sample:")
    df_silver.show(5, truncate=False)
    print(f"\n[INFO] Silver Parquet saved to: {OUTPUT_DIR}\n")

    spark.stop()
