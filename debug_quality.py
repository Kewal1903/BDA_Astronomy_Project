#!/usr/bin/env python3

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize Spark
spark = SparkSession.builder \
    .appName("Quality-Debug") \
    .config("spark.master", "local[*]") \
    .getOrCreate()

# Read the astronomical analysis results
df = spark.read.parquet("hdfs://namenode:8020/user/spark/gold/astronomical_analysis_20251012_153354")

print("=== QUALITY SCORE DEBUGGING ===")

# Check for NaN values by image type
print("\n1. Quality Score Statistics by Image Type:")
quality_stats = df.groupBy("image_type") \
    .agg(
        count("*").alias("total_count"),
        sum(when(isnan(col("quality_score")) | col("quality_score").isNull(), 1).otherwise(0)).alias("nan_count"),
        avg(when(~isnan(col("quality_score")) & col("quality_score").isNotNull(), col("quality_score"))).alias("avg_quality_non_nan"),
        min("quality_score").alias("min_quality"),
        max("quality_score").alias("max_quality")
    ) \
    .orderBy("image_type")

quality_stats.show()

# Sample records with NaN quality scores
print("\n2. Sample TEMPLATE records with NaN quality:")
template_nan = df.filter(
    (col("image_type") == "TEMPLATE") & 
    (isnan(col("quality_score")) | col("quality_score").isNull())
).select(
    "file_path", "quality_score", "dynamic_range", "background_noise", 
    "total_objects", "saturation_fraction", "processing_status"
).limit(5)

template_nan.show(truncate=False)

# Compare with non-NaN template records
print("\n3. Sample TEMPLATE records with valid quality:")
template_valid = df.filter(
    (col("image_type") == "TEMPLATE") & 
    (~isnan(col("quality_score")) & col("quality_score").isNotNull())
).select(
    "file_path", "quality_score", "dynamic_range", "background_noise", 
    "total_objects", "saturation_fraction", "processing_status"
).limit(5)

template_valid.show(truncate=False)

# Check for specific conditions that might cause NaN
print("\n4. Condition Analysis for TEMPLATE images:")
template_analysis = df.filter(col("image_type") == "TEMPLATE") \
    .agg(
        sum(when(col("background_noise") <= 0, 1).otherwise(0)).alias("zero_background_noise"),
        sum(when(col("dynamic_range") <= 0, 1).otherwise(0)).alias("zero_dynamic_range"),
        sum(when(col("total_objects") == 0, 1).otherwise(0)).alias("zero_objects"),
        sum(when(col("saturation_fraction").isNull(), 1).otherwise(0)).alias("null_saturation"),
    )

template_analysis.show()

spark.stop()