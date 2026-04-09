# app/adaptive_resource_manager.py
"""
Adaptive Resource Management for Astronomical Processing
Implements resource-aware computation strategies for variable data loads
"""

import psutil
import time
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum as spark_sum, max as spark_max
import numpy as np

class ResourceMonitor:
    """Monitor system resources and provide optimization recommendations"""
    
    def __init__(self):
        self.metrics_history = []
        self.performance_baselines = {}
    
    def collect_metrics(self):
        """Collect current system metrics"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_io_read_mb_s': 0,  # Simplified
            'disk_io_write_mb_s': 0,  # Simplified
            'network_io_mb_s': 0  # Simplified
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 measurements
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def get_resource_utilization(self):
        """Get current resource utilization summary"""
        if not self.metrics_history:
            return None
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            'avg_cpu_percent': np.mean([m['cpu_percent'] for m in recent_metrics]),
            'avg_memory_percent': np.mean([m['memory_percent'] for m in recent_metrics]),
            'min_memory_available_gb': min([m['memory_available_gb'] for m in recent_metrics]),
            'resource_pressure': self._calculate_resource_pressure(recent_metrics)
        }
    
    def _calculate_resource_pressure(self, metrics):
        """Calculate overall resource pressure (0=low, 1=high)"""
        cpu_pressure = np.mean([m['cpu_percent'] for m in metrics]) / 100.0
        memory_pressure = np.mean([m['memory_percent'] for m in metrics]) / 100.0
        
        # Weighted combination
        overall_pressure = 0.6 * cpu_pressure + 0.4 * memory_pressure
        return min(overall_pressure, 1.0)
    
    def recommend_spark_config(self, data_size_gb, num_files):
        """Recommend Spark configuration based on current resources"""
        utilization = self.get_resource_utilization()
        
        if not utilization:
            # Default configuration
            return {
                'executor_memory': '1g',
                'executor_cores': 1,
                'num_partitions': 4,
                'adaptive_enabled': True
            }
        
        available_memory_gb = utilization['min_memory_available_gb']
        resource_pressure = utilization['resource_pressure']
        
        # Adaptive configuration based on resources
        if resource_pressure < 0.3:  # Low pressure
            executor_memory = min(int(available_memory_gb * 0.4), 2)
            executor_cores = 2
            num_partitions = max(8, min(num_files // 10, 16))
        elif resource_pressure < 0.7:  # Medium pressure
            executor_memory = min(int(available_memory_gb * 0.3), 1)
            executor_cores = 1
            num_partitions = max(4, min(num_files // 20, 8))
        else:  # High pressure
            executor_memory = 1
            executor_cores = 1
            num_partitions = max(2, min(num_files // 50, 4))
        
        return {
            'executor_memory': f'{executor_memory}g',
            'executor_cores': executor_cores,
            'num_partitions': num_partitions,
            'adaptive_enabled': True,
            'resource_pressure': resource_pressure
        }

class AdaptiveProcessingPipeline:
    """Adaptive processing pipeline that adjusts to system resources"""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.processing_history = []
    
    def create_adaptive_spark_session(self, config):
        """Create Spark session with adaptive configuration"""
        builder = SparkSession.builder.appName("Adaptive-Astronomical-Processing")
        
        # Apply resource-aware configuration
        builder = builder.config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        builder = builder.config("spark.executor.memory", config['executor_memory'])
        builder = builder.config("spark.executor.cores", str(config['executor_cores']))
        builder = builder.config("spark.sql.adaptive.enabled", str(config['adaptive_enabled']).lower())
        builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        builder = builder.config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
        
        # Dynamic allocation settings
        builder = builder.config("spark.dynamicAllocation.enabled", "false")  # Simplified for single worker
        
        return builder.getOrCreate()
    
    def estimate_processing_time(self, num_files, avg_file_size_mb, config):
        """Estimate processing time based on historical data and current config"""
        
        if not self.processing_history:
            # Default estimate
            base_time_per_file = 2.0  # seconds
            return num_files * base_time_per_file
        
        # Use historical performance data
        recent_jobs = self.processing_history[-5:]  # Last 5 jobs
        
        if recent_jobs:
            avg_files_per_second = np.mean([
                job['files_processed'] / job['execution_time_seconds'] 
                for job in recent_jobs if job['execution_time_seconds'] > 0
            ])
            
            if avg_files_per_second > 0:
                estimated_time = num_files / avg_files_per_second
                
                # Adjust for resource pressure
                pressure_factor = 1.0 + config.get('resource_pressure', 0) * 0.5
                return estimated_time * pressure_factor
        
        return num_files * 2.0  # Fallback estimate
    
    def process_with_adaptive_strategy(self, input_path, processing_function, output_path):
        """Execute processing with adaptive resource management"""
        
        print("=== Adaptive Resource Management Pipeline ===")
        
        # Initial resource assessment
        initial_metrics = self.resource_monitor.collect_metrics()
        print(f"Initial CPU: {initial_metrics['cpu_percent']:.1f}%, "
              f"Memory: {initial_metrics['memory_percent']:.1f}%")
        
        # Estimate data characteristics
        temp_spark = SparkSession.builder.appName("DataAssessment").getOrCreate()
        try:
            files_rdd = temp_spark.sparkContext.binaryFiles(input_path)
            num_files = files_rdd.count()
            avg_file_size = files_rdd.map(lambda x: len(x[1])).mean() / (1024*1024)  # MB
            total_size_gb = (num_files * avg_file_size) / 1024
            
            print(f"Dataset assessment: {num_files} files, "
                  f"avg size: {avg_file_size:.1f}MB, "
                  f"total: {total_size_gb:.2f}GB")
        except Exception as e:
            print(f"Could not assess dataset: {e}")
            num_files, avg_file_size, total_size_gb = 100, 20, 2  # Defaults
        finally:
            temp_spark.stop()
        
        # Get adaptive configuration
        config = self.resource_monitor.recommend_spark_config(total_size_gb, num_files)
        print(f"Adaptive config: memory={config['executor_memory']}, "
              f"cores={config['executor_cores']}, "
              f"partitions={config['num_partitions']}")
        
        # Estimate processing time
        estimated_time = self.estimate_processing_time(num_files, avg_file_size, config)
        print(f"Estimated processing time: {estimated_time:.1f} seconds")
        
        # Create adaptive Spark session
        spark = self.create_adaptive_spark_session(config)
        
        try:
            start_time = time.time()
            
            # Monitor resources during processing
            monitoring_thread = None  # Simplified - would use threading in production
            
            # Execute processing function with adaptive parameters
            result = processing_function(spark, input_path, output_path, config)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Collect final metrics
            final_metrics = self.resource_monitor.collect_metrics()
            
            # Record performance for future optimization
            performance_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'num_files': num_files,
                'total_size_gb': total_size_gb,
                'config': config,
                'execution_time_seconds': execution_time,
                'files_processed': num_files,
                'throughput_files_per_second': num_files / execution_time if execution_time > 0 else 0,
                'initial_cpu_percent': initial_metrics['cpu_percent'],
                'final_cpu_percent': final_metrics['cpu_percent'],
                'initial_memory_percent': initial_metrics['memory_percent'],
                'final_memory_percent': final_metrics['memory_percent']
            }
            
            self.processing_history.append(performance_record)
            
            print(f"\n=== PERFORMANCE SUMMARY ===")
            print(f"Execution time: {execution_time:.1f}s "
                  f"(estimated: {estimated_time:.1f}s)")
            print(f"Throughput: {performance_record['throughput_files_per_second']:.2f} files/second")
            print(f"Resource efficiency: CPU {initial_metrics['cpu_percent']:.1f}% → "
                  f"{final_metrics['cpu_percent']:.1f}%, "
                  f"Memory {initial_metrics['memory_percent']:.1f}% → "
                  f"{final_metrics['memory_percent']:.1f}%")
            
            return result, performance_record
            
        finally:
            spark.stop()

def adaptive_fits_processing(spark, input_path, output_path, config):
    """Adaptive FITS processing function"""
    
    print(f"Processing FITS files with adaptive strategy...")
    
    # Load data with adaptive partitioning
    fits_rdd = spark.sparkContext.binaryFiles(input_path, 
                                             minPartitions=config['num_partitions'])
    
    def lightweight_fits_analysis(records):
        """Lightweight analysis adapted to resource constraints"""
        import io
        import numpy as np
        from astropy.io import fits
        import warnings
        warnings.simplefilter("ignore")
        
        results = []
        for path, data in records:
            try:
                with fits.open(io.BytesIO(data), memmap=False) as hdul:
                    hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                    if hdu is None or hdu.data is None:
                        results.append((path, "NO_DATA", 0, 0, 0))
                        continue
                    
                    img = np.asarray(hdu.data, dtype=np.float32)
                    
                    # Adaptive analysis based on resource pressure
                    resource_pressure = config.get('resource_pressure', 0)
                    
                    if resource_pressure < 0.3:
                        # Full analysis for low pressure
                        valid_data = img[np.isfinite(img)]
                        if len(valid_data) > 0:
                            mean_val = float(np.mean(valid_data))
                            std_val = float(np.std(valid_data))
                            num_objects = int(np.sum(valid_data > mean_val + 3*std_val))
                        else:
                            mean_val = std_val = num_objects = 0
                    else:
                        # Simplified analysis for high pressure
                        # Sample subset of pixels for faster processing
                        sample_size = min(10000, img.size)
                        sample_indices = np.random.choice(img.size, sample_size, replace=False)
                        sample_data = img.flat[sample_indices]
                        sample_data = sample_data[np.isfinite(sample_data)]
                        
                        if len(sample_data) > 0:
                            mean_val = float(np.mean(sample_data))
                            std_val = float(np.std(sample_data))
                            # Estimate objects based on sample
                            num_objects = int(np.sum(sample_data > mean_val + 3*std_val) * 
                                            (img.size / sample_size))
                        else:
                            mean_val = std_val = num_objects = 0
                    
                    results.append((path, "SUCCESS", mean_val, std_val, num_objects))
                    
            except Exception as e:
                results.append((path, f"ERROR_{str(e)[:20]}", 0, 0, 0))
        
        return iter(results)
    
    # Process with adaptive strategy
    processed_rdd = fits_rdd.mapPartitions(lightweight_fits_analysis)
    
    # Adaptive output strategy
    if config['resource_pressure'] < 0.5:
        # Use more partitions for better parallelism
        output_partitions = config['num_partitions']
    else:
        # Reduce partitions to save resources
        output_partitions = max(1, config['num_partitions'] // 2)
    
    # Save results
    processed_rdd.coalesce(output_partitions).map(
        lambda x: f"{x[0]},{x[1]},{x[2]:.4f},{x[3]:.4f},{x[4]}"
    ).saveAsTextFile(output_path)
    
    return processed_rdd.count()

def main():
    """Main execution with adaptive resource management"""
    
    pipeline = AdaptiveProcessingPipeline()
    
    # Input and output paths
    input_path = "hdfs://namenode:8020/user/astro/raw/ztf_fits/*.fits"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = f"hdfs://namenode:8020/user/spark/adaptive_results_{timestamp}"
    
    # Execute adaptive processing
    try:
        result, performance = pipeline.process_with_adaptive_strategy(
            input_path, 
            adaptive_fits_processing, 
            output_path
        )
        
        print(f"\nAdaptive processing completed successfully!")
        print(f"Results saved to: {output_path}")
        
        # Save performance metrics
        metrics_path = f"hdfs://namenode:8020/user/spark/performance_metrics_{timestamp}.json"
        
        # Create a simple Spark session for saving metrics
        spark = SparkSession.builder.appName("SaveMetrics").getOrCreate()
        spark.sparkContext.parallelize([json.dumps(performance, indent=2)]).coalesce(1).saveAsTextFile(metrics_path)
        spark.stop()
        
        print(f"Performance metrics saved to: {metrics_path}")
        
    except Exception as e:
        print(f"Adaptive processing failed: {e}")

if __name__ == "__main__":
    main()