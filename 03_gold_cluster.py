from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

if __name__ == "__main__":
    # Initialize Spark for YARN Cluster processing
    spark = SparkSession.builder \
        .appName("03_Gold_Clustering_Cluster") \
        .getOrCreate()

    # SILENCE THE NOISE: Only show Warnings and Errors
    spark.sparkContext.setLogLevel("WARN")

    # Define HDFS paths linking Silver input to Gold output
    INPUT_DIR = "hdfs://master:9000/user/ztf/subset_12k_silver.parquet"
    OUTPUT_DIR = "hdfs://master:9000/user/ztf/subset_12k_gold.parquet"

    print(f"\n[SYSTEM] Loading Silver Features from: {INPUT_DIR}")
    # 1. Load the Silver data
    df_silver = spark.read.parquet(INPUT_DIR)

    # 2. Select the features that indicate something changed in the sky
    feature_cols = [
        "DIFFERENCE_std_sub", 
        "DIFFERENCE_bright_pixels",
        "SCIENCE_std_sub"
    ]

    # 3. Assemble them into a single Vector column
    print("[SYSTEM] Vectorizing and Scaling features...")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    df_vectorized = assembler.transform(df_silver)

    # 4. Scale the data (Standardization is crucial for K-Means)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_vectorized)
    df_scaled = scaler_model.transform(df_vectorized)

    # 5. Initialize and train the distributed K-Means algorithm
    print("[SYSTEM] Training Distributed K-Means Model (k=3)...")
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=3, seed=42)
    model = kmeans.fit(df_scaled)

    # 6. Generate the predictions (Assign every object to a cluster)
    df_predictions = model.transform(df_scaled)

    # 7. Evaluate the model mathematically
    evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="features", metricName="silhouette")
    silhouette = evaluator.evaluate(df_predictions)

    print(f"\n[SUCCESS] Model Training Complete!")
    print(f"[INFO] Silhouette Score (Quality): {silhouette:.4f}\n")

    # 8. See how many objects fell into each cluster
    print("[SYSTEM] Astronomical Object Clusters Distribution:")
    df_predictions.groupBy("cluster").count().orderBy("cluster").show()

    # 9. Save the Gold ML Output
    print("\n[SYSTEM] Saving Final Gold Machine Learning Output...")
    df_predictions.write.mode("overwrite").parquet(OUTPUT_DIR)
    print(f"[SUCCESS] Gold Layer complete! Data safely written to: {OUTPUT_DIR}\n")

    # 10. Shut down Spark
    spark.stop()
