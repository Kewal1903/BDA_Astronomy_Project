from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ReadFromHDFS") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
    .getOrCreate()

df = spark.read.csv("hdfs://namenode:8020/user/spark/input/sample.csv", header=True, inferSchema=True)
df.show()

output_path = "hdfs://namenode:8020/user/spark/output/result.csv"

df.write.mode("overwrite").csv(output_path, header=True)

print("✅ Output written to:", output_path)
out = "hdfs://namenode:8020/user/spark/output/result_single.csv"
df.coalesce(1).write.mode("overwrite").csv(out, header=True)
print("Wrote:", out)


spark.stop()
