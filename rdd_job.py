from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("RDD-Minimum")
sc = SparkContext(conf=conf)

# Read raw lines from HDFS (treat CSV as text to use pure RDD ops)
lines = sc.textFile("hdfs://namenode:8020/user/spark/input/sample.csv")

# Transformations (lazy)
header = lines.first()
data = lines.filter(lambda x: x != header).filter(lambda x: x.strip() != "")
pairs = data.map(lambda row: row.split(","))            # ["Alice","25"]
ages  = pairs.map(lambda cols: int(cols[1]))

# Actions (materialize)
count = ages.count()
total = ages.reduce(lambda a,b: a+b)
max_age = ages.max()
min_age = ages.min()
avg = total / count if count else 0

print("Count:", count, "Sum:", total, "Avg:", avg, "Min:", min_age, "Max:", max_age)

# Save a tiny summary back to HDFS (as text)
summary = sc.parallelize([f"count={count},sum={total},avg={avg},min={min_age},max={max_age}"])
summary.saveAsTextFile("hdfs://namenode:8020/user/spark/output/rdd_summary")

sc.stop()
