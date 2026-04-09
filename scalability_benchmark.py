# app/scalability_benchmark.py
"""
Scalability Benchmark for Astronomical Processing
Demonstrates horizontal scaling capabilities and performance characteristics
"""

import time
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum as spark_sum, max as spark_max, min as spark_min
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
import numpy as np

class ScalabilityBenchmark:
    """Benchmark suite for testing scalability characteristics"""
    
    def __init__(self):
        self.benchmark_results = []
        self.scaling_metrics = {}
    
    def create_benchmark_spark_session(self, config_name, memory_per_executor="1g", num_cores=1):
        """Create Spark session with specific configuration for benchmarking"""
        
        builder = SparkSession.builder.appName(f"Scalability-Benchmark-{config_name}")
        
        # Core configuration
        builder = builder.config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        builder = builder.config("spark.executor.memory", memory_per_executor)
        builder = builder.config("spark.executor.cores", str(num_cores))
        
        # Performance tuning
        builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        builder = builder.config("spark.sql.adaptive.enabled", "true")
        builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        builder = builder.config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
        
        # Memory management
        builder = builder.config("spark.executor.memoryFraction", "0.8")
        builder = builder.config("spark.storage.memoryFraction", "0.5")
        
        return builder.getOrCreate()
    
    def synthetic_data_generation_benchmark(self, num_files_list, output_base_path):
        """Generate synthetic astronomical data for scaling tests"""
        
        print("=== SYNTHETIC DATA GENERATION BENCHMARK ===")
        
        for num_files in num_files_list:
            print(f"\nGenerating {num_files} synthetic FITS files...")
            
            spark = self.create_benchmark_spark_session(f"DataGen-{num_files}")
            
            try:
                start_time = time.time()
                
                # Generate synthetic astronomical data
                def generate_synthetic_fits_data(partition_id):
                    """Generate synthetic FITS-like data"""
                    import numpy as np
                    import struct
                    
                    results = []
                    files_per_partition = max(1, num_files // 4)  # Distribute across partitions
                    
                    for i in range(files_per_partition):
                        file_id = partition_id * files_per_partition + i
                        if file_id >= num_files:
                            break
                        
                        # Generate synthetic image data (100x100 pixels)
                        np.random.seed(file_id)  # Reproducible data
                        
                        # Simulate astronomical image with background + sources
                        background = np.random.normal(1000, 50, (100, 100))
                        
                        # Add synthetic stars/galaxies
                        num_sources = np.random.poisson(20)
                        for _ in range(num_sources):
                            x, y = np.random.randint(10, 90, 2)
                            brightness = np.random.exponential(5000)
                            sigma = np.random.uniform(1, 3)
                            
                            # Add Gaussian sources
                            xx, yy = np.meshgrid(range(100), range(100))
                            distance = np.sqrt((xx - x)**2 + (yy - y)**2)
                            source = brightness * np.exp(-0.5 * (distance / sigma)**2)
                            background += source
                        
                        # Add noise
                        noise = np.random.normal(0, np.sqrt(background))
                        final_image = background + noise
                        
                        # Calculate statistics for this synthetic image
                        mean_val = float(np.mean(final_image))
                        std_val = float(np.std(final_image))
                        max_val = float(np.max(final_image))
                        num_bright = int(np.sum(final_image > mean_val + 5*std_val))
                        
                        # Create synthetic metadata
                        filepath = f"synthetic_fits_{file_id:06d}.fits"
                        
                        results.append((
                            filepath, mean_val, std_val, max_val, num_bright,
                            datetime.utcnow().isoformat(), 100*100, "SYNTHETIC"
                        ))
                    
                    return iter(results)
                
                # Create RDD with synthetic data
                num_partitions = min(4, num_files)
                synthetic_rdd = spark.sparkContext.range(num_partitions).mapPartitions(
                    generate_synthetic_fits_data
                )
                
                # Convert to DataFrame for better performance analysis
                schema = StructType([
                    StructField("filepath", StringType(), True),
                    StructField("mean_value", DoubleType(), True),
                    StructField("std_value", DoubleType(), True),
                    StructField("max_value", DoubleType(), True),
                    StructField("bright_objects", IntegerType(), True),
                    StructField("timestamp", StringType(), True),
                    StructField("total_pixels", IntegerType(), True),
                    StructField("data_type", StringType(), True)
                ])
                
                synthetic_df = spark.createDataFrame(synthetic_rdd, schema)
                
                # Save synthetic data
                output_path = f"{output_base_path}/synthetic_{num_files}_files"
                synthetic_df.coalesce(2).write.mode("overwrite").parquet(output_path)
                
                generation_time = time.time() - start_time
                
                print(f"Generated {num_files} files in {generation_time:.2f}s "
                      f"({num_files/generation_time:.1f} files/sec)")
                
                # Record benchmark result
                self.benchmark_results.append({
                    'benchmark_type': 'data_generation',
                    'num_files': num_files,
                    'execution_time': generation_time,
                    'throughput_files_per_sec': num_files / generation_time,
                    'output_path': output_path
                })
                
            finally:
                spark.stop()
    
    def processing_scalability_benchmark(self, data_paths, partition_configs):
        """Test processing scalability with different partition configurations"""
        
        print("\n=== PROCESSING SCALABILITY BENCHMARK ===")
        
        for data_path in data_paths:
            for config in partition_configs:
                config_name = f"P{config['partitions']}_M{config['memory']}"
                print(f"\nTesting configuration: {config_name}")
                print(f"Data path: {data_path}")
                
                spark = self.create_benchmark_spark_session(
                    config_name, 
                    config['memory'], 
                    config.get('cores', 1)
                )
                
                try:
                    start_time = time.time()
                    
                    # Load data
                    df = spark.read.parquet(data_path)
                    initial_count = df.count()
                    
                    print(f"Loaded {initial_count} records")
                    
                    # Repartition according to config
                    if config['partitions'] != df.rdd.getNumPartitions():
                        df = df.repartition(config['partitions'])
                    
                    # Perform complex astronomical analysis
                    analysis_results = df.select(
                        count("*").alias("total_files"),
                        avg("mean_value").alias("avg_background"),
                        avg("std_value").alias("avg_noise"),
                        spark_max("max_value").alias("brightest_pixel"),
                        spark_sum("bright_objects").alias("total_bright_objects"),
                        avg("bright_objects").alias("avg_objects_per_image")
                    ).collect()[0]
                    
                    # Advanced aggregations
                    quality_metrics = df.select(
                        col("mean_value"),
                        col("std_value"),
                        col("bright_objects")
                    ).rdd.map(lambda row: {
                        'snr': row['mean_value'] / row['std_value'] if row['std_value'] > 0 else 0,
                        'object_density': row['bright_objects'] / 10000,  # objects per 10k pixels
                        'quality_score': min(100, row['mean_value'] / 10 + 
                                           (row['bright_objects'] * 5) - 
                                           (row['std_value'] / 50))
                    }).collect()
                    
                    avg_snr = np.mean([m['snr'] for m in quality_metrics])
                    avg_object_density = np.mean([m['object_density'] for m in quality_metrics])
                    avg_quality_score = np.mean([m['quality_score'] for m in quality_metrics])
                    
                    processing_time = time.time() - start_time
                    throughput = initial_count / processing_time
                    
                    print(f"Processing completed in {processing_time:.2f}s")
                    print(f"Throughput: {throughput:.1f} files/sec")
                    print(f"Background: {analysis_results['avg_background']:.1f} ± "
                          f"{analysis_results['avg_noise']:.1f}")
                    print(f"Objects per image: {analysis_results['avg_objects_per_image']:.1f}")
                    print(f"Average SNR: {avg_snr:.2f}")
                    print(f"Quality score: {avg_quality_score:.1f}")
                    
                    # Record detailed benchmark result
                    benchmark_result = {
                        'benchmark_type': 'processing_scalability',
                        'data_path': data_path,
                        'configuration': config,
                        'num_files': initial_count,
                        'execution_time': processing_time,
                        'throughput_files_per_sec': throughput,
                        'num_partitions_used': df.rdd.getNumPartitions(),
                        'analysis_results': {
                            'total_files': analysis_results['total_files'],
                            'avg_background': analysis_results['avg_background'],
                            'avg_noise': analysis_results['avg_noise'],
                            'brightest_pixel': analysis_results['brightest_pixel'],
                            'total_bright_objects': analysis_results['total_bright_objects'],
                            'avg_objects_per_image': analysis_results['avg_objects_per_image'],
                            'avg_snr': avg_snr,
                            'avg_object_density': avg_object_density,
                            'avg_quality_score': avg_quality_score
                        }
                    }
                    
                    self.benchmark_results.append(benchmark_result)
                    
                except Exception as e:
                    print(f"Configuration {config_name} failed: {e}")
                    
                finally:
                    spark.stop()
                    time.sleep(2)  # Cool-down period
    
    def analyze_scaling_characteristics(self):
        """Analyze scaling characteristics from benchmark results"""
        
        print("\n=== SCALING CHARACTERISTICS ANALYSIS ===")
        
        processing_results = [r for r in self.benchmark_results 
                            if r['benchmark_type'] == 'processing_scalability']
        
        if not processing_results:
            print("No processing results to analyze")
            return
        
        # Group by data size (number of files)
        size_groups = {}
        for result in processing_results:
            size = result['num_files']
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(result)
        
        print(f"Analyzing {len(processing_results)} benchmark runs across {len(size_groups)} data sizes")
        
        scaling_analysis = {}
        
        for size, results in size_groups.items():
            if len(results) < 2:
                continue
            
            # Analyze how different configurations performed on same data size
            best_config = max(results, key=lambda x: x['throughput_files_per_sec'])
            worst_config = min(results, key=lambda x: x['throughput_files_per_sec'])
            
            throughputs = [r['throughput_files_per_sec'] for r in results]
            execution_times = [r['execution_time'] for r in results]
            
            scaling_analysis[size] = {
                'num_configurations_tested': len(results),
                'best_throughput': best_config['throughput_files_per_sec'],
                'worst_throughput': worst_config['throughput_files_per_sec'],
                'best_config': best_config['configuration'],
                'worst_config': worst_config['configuration'],
                'throughput_variance': np.var(throughputs),
                'avg_execution_time': np.mean(execution_times),
                'speedup_factor': best_config['throughput_files_per_sec'] / worst_config['throughput_files_per_sec']
            }
            
            print(f"\nData size: {size} files")
            print(f"  Best throughput: {best_config['throughput_files_per_sec']:.1f} files/sec "
                  f"(config: {best_config['configuration']})")
            print(f"  Worst throughput: {worst_config['throughput_files_per_sec']:.1f} files/sec "
                  f"(config: {worst_config['configuration']})")
            print(f"  Speedup factor: {scaling_analysis[size]['speedup_factor']:.2f}x")
        
        # Overall scalability insights
        if len(scaling_analysis) >= 2:
            sizes = sorted(scaling_analysis.keys())
            linear_scaling_efficiency = []
            
            for i in range(1, len(sizes)):
                prev_size = sizes[i-1]
                curr_size = sizes[i]
                
                prev_best = scaling_analysis[prev_size]['best_throughput']
                curr_best = scaling_analysis[curr_size]['best_throughput']
                
                expected_scaling = curr_size / prev_size
                actual_scaling = curr_best / prev_best
                efficiency = actual_scaling / expected_scaling
                
                linear_scaling_efficiency.append(efficiency)
            
            avg_efficiency = np.mean(linear_scaling_efficiency)
            
            print(f"\n=== OVERALL SCALING INSIGHTS ===")
            print(f"Average linear scaling efficiency: {avg_efficiency:.2%}")
            
            if avg_efficiency > 0.8:
                print("✅ Excellent horizontal scaling characteristics")
            elif avg_efficiency > 0.6:
                print("✅ Good horizontal scaling with room for optimization")
            else:
                print("⚠️  Sublinear scaling - resource bottlenecks detected")
        
        self.scaling_metrics = scaling_analysis
        
        return scaling_analysis
    
    def save_benchmark_results(self, output_path):
        """Save complete benchmark results"""
        
        complete_results = {
            'benchmark_metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'total_benchmarks': len(self.benchmark_results),
                'scaling_metrics': self.scaling_metrics
            },
            'individual_results': self.benchmark_results,
            'scaling_analysis': self.scaling_metrics
        }
        
        # Save using Spark for HDFS compatibility
        spark = SparkSession.builder.appName("SaveBenchmarks").getOrCreate()
        
        try:
            results_json = json.dumps(complete_results, indent=2, default=str)
            spark.sparkContext.parallelize([results_json]).coalesce(1).saveAsTextFile(output_path)
            print(f"Benchmark results saved to: {output_path}")
        finally:
            spark.stop()

def main():
    """Main scalability benchmark execution"""
    
    print("=== ASTRONOMICAL PROCESSING SCALABILITY BENCHMARK ===")
    print(f"Start time: {datetime.utcnow().isoformat()}")
    
    benchmark = ScalabilityBenchmark()
    
    # Configuration for scalability tests
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_output_path = "hdfs://namenode:8020/user/spark/scalability_test"
    
    # Test different data sizes
    file_counts = [100, 500, 1000]  # Scaled for single worker
    
    # Test different processing configurations
    partition_configs = [
        {'partitions': 2, 'memory': '512m', 'cores': 1},
        {'partitions': 4, 'memory': '1g', 'cores': 1},
        {'partitions': 8, 'memory': '1g', 'cores': 2},
        {'partitions': 1, 'memory': '2g', 'cores': 1}  # Single partition test
    ]
    
    try:
        # Step 1: Generate synthetic data for scalability testing
        print("Phase 1: Generating synthetic astronomical data...")
        benchmark.synthetic_data_generation_benchmark(
            file_counts, 
            f"{base_output_path}/synthetic_data"
        )
        
        # Step 2: Test processing scalability
        print("\nPhase 2: Testing processing scalability...")
        data_paths = [
            f"{base_output_path}/synthetic_data/synthetic_{count}_files" 
            for count in file_counts
        ]
        
        benchmark.processing_scalability_benchmark(data_paths, partition_configs)
        
        # Step 3: Analyze scaling characteristics
        print("\nPhase 3: Analyzing scaling characteristics...")
        scaling_analysis = benchmark.analyze_scaling_characteristics()
        
        # Step 4: Save comprehensive results
        results_path = f"{base_output_path}/benchmark_results_{timestamp}"
        benchmark.save_benchmark_results(results_path)
        
        print(f"\n=== SCALABILITY BENCHMARK COMPLETED ===")
        print(f"Total benchmark runs: {len(benchmark.benchmark_results)}")
        print(f"Results saved to: {results_path}")
        print(f"End time: {datetime.utcnow().isoformat()}")
        
    except Exception as e:
        print(f"Scalability benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()