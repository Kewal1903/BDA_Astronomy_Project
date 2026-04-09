#!/usr/bin/env python3
"""
Long-Running Astronomical Data Monitor
This application will stay active in Spark UI for demonstration
"""

from pyspark.sql import SparkSession
import time
import json
from datetime import datetime

def main():
    # Create Spark session with visible application name
    spark = SparkSession.builder \
        .appName("🔭 Astronomical-Data-Monitor-LIVE") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
        .config("spark.executor.memory", "1g") \
        .config("spark.executor.cores", "1") \
        .getOrCreate()
    
    print("=" * 60)
    print("🚀 LONG-RUNNING ASTRONOMICAL MONITOR STARTED")
    print("=" * 60)
    print("📊 Check Spark UI at: http://localhost:4040")
    print("🔍 Application Name: 🔭 Astronomical-Data-Monitor-LIVE")
    print("⏱️  This will run for 5 minutes to demonstrate Spark UI")
    print("=" * 60)
    
    # Monitor astronomical data processing for 5 minutes
    cycles = 30  # 30 cycles of 10 seconds each = 5 minutes
    
    for cycle in range(1, cycles + 1):
        print(f"\n🔄 Monitoring Cycle {cycle}/{cycles} - {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Read bronze layer statistics
            bronze_df = spark.read.text("hdfs://namenode:8020/user/spark/bronze/fits_stats_full*/part-*")
            bronze_count = bronze_df.count()
            
            # Read gold layer results if available
            try:
                gold_df = spark.read.parquet("hdfs://namenode:8020/user/spark/gold/astronomical_analysis_*/")
                gold_count = gold_df.count()
                
                # Calculate some live statistics
                avg_objects = gold_df.agg({"total_objects": "avg"}).collect()[0][0]
                max_quality = gold_df.agg({"quality_score": "max"}).collect()[0][0]
                
                print(f"📈 Live Stats: {bronze_count} processed files, {gold_count} analyzed objects")
                print(f"🌟 Avg objects/image: {avg_objects:.1f}, Best quality: {max_quality:.3f}")
                
            except Exception:
                print(f"📊 Bronze layer: {bronze_count} files processed")
                print("⏳ Gold layer analysis pending...")
            
            # Simulate some computational work to keep Spark busy
            dummy_rdd = spark.sparkContext.parallelize(range(1000), 4)
            result = dummy_rdd.map(lambda x: x * x).filter(lambda x: x % 2 == 0).count()
            print(f"🔧 Background computation result: {result}")
            
        except Exception as e:
            print(f"⚠️  Error accessing data: {e}")
            print("🔄 Continuing monitoring...")
        
        # Show progress bar
        progress = "█" * (cycle * 20 // cycles) + "░" * (20 - cycle * 20 // cycles)
        print(f"📊 Progress: [{progress}] {cycle*100//cycles:3d}%")
        
        if cycle < cycles:
            print("⏱️  Sleeping 10 seconds... (Spark application remains active)")
            time.sleep(10)
    
    print("\n" + "=" * 60)
    print("✅ ASTRONOMICAL MONITORING COMPLETED")
    print("🏁 Spark application will now terminate and disappear from UI")
    print("=" * 60)
    
    spark.stop()

if __name__ == "__main__":
    main()