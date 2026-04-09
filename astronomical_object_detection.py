# app/astronomical_object_detection.py
"""
Advanced Astronomical Object Detection Pipeline
Implements sophisticated analysis for celestial object identification and characterization
"""

import io
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, stddev, max as spark_max, min as spark_min
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType
import json
from datetime import datetime

# Initialize Spark with optimized configuration for astronomical processing
spark = (
    SparkSession.builder
    .appName("Astronomical-Object-Detection")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
    .config("spark.executor.memory", "1g")
    .config("spark.executor.cores", "1")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)
sc = spark.sparkContext

def advanced_fits_analysis(records):
    """
    Advanced FITS analysis including:
    - Object detection using statistical thresholding
    - Source extraction and characterization
    - Photometric measurements
    - Quality assessment metrics
    """
    import io, warnings
    import numpy as np
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyWarning
    from scipy import ndimage
    from skimage.measure import label, regionprops
    warnings.simplefilter("ignore", VerifyWarning)

    results = []
    
    for path, data in records:
        try:
            analysis_result = {
                'file_path': path,
                'telescope_id': extract_telescope_id(path),
                'observation_time': extract_observation_time(path),
                'image_type': extract_image_type(path),
                'total_objects': 0,
                'bright_sources': 0,
                'faint_sources': 0,
                'background_level': 0.0,
                'background_noise': 0.0,
                'dynamic_range': 0.0,
                'saturation_fraction': 0.0,
                'source_positions': [],
                'photometry_data': [],
                'quality_score': 0.0,
                'processing_status': 'SUCCESS'
            }
            
            with fits.open(io.BytesIO(data), memmap=False) as hdul:
                # Find the primary image HDU
                hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                if hdu is None or hdu.data is None:
                    analysis_result['processing_status'] = 'NO_IMAGE_DATA'
                    results.append(analysis_result)
                    continue
                
                img = np.asarray(hdu.data, dtype=np.float32)
                img = np.where(np.isfinite(img), img, np.nan)
                
                if np.isnan(img).all():
                    analysis_result['processing_status'] = 'ALL_NAN'
                    results.append(analysis_result)
                    continue
                
                # Background estimation and noise characterization
                valid_pixels = img[np.isfinite(img)]
                if len(valid_pixels) == 0:
                    analysis_result['processing_status'] = 'NO_VALID_PIXELS'
                    results.append(analysis_result)
                    continue
                
                # Robust background estimation using sigma-clipped statistics
                background_level = np.nanmedian(valid_pixels)
                mad = np.nanmedian(np.abs(valid_pixels - background_level))
                background_noise = 1.4826 * mad  # Convert MAD to standard deviation
                
                analysis_result['background_level'] = float(background_level)
                analysis_result['background_noise'] = float(background_noise)
                
                # Dynamic range calculation
                min_val, max_val = np.nanmin(valid_pixels), np.nanmax(valid_pixels)
                analysis_result['dynamic_range'] = float(max_val - min_val)
                
                # Saturation detection (assuming 16-bit images)
                saturation_threshold = 65000  # Typical for 16-bit
                saturated_pixels = np.sum(valid_pixels >= saturation_threshold)
                analysis_result['saturation_fraction'] = float(saturated_pixels / len(valid_pixels))
                
                # Object detection using multiple sigma thresholds
                if background_noise > 0:
                    # Create detection thresholds
                    detection_threshold = background_level + 5.0 * background_noise
                    bright_threshold = background_level + 10.0 * background_noise
                    
                    # Background-subtracted image
                    img_sub = img - background_level
                    
                    # Create binary masks for detection
                    detection_mask = img_sub > (5.0 * background_noise)
                    bright_mask = img_sub > (10.0 * background_noise)
                    
                    # Label connected components (sources)
                    labeled_sources = label(detection_mask)
                    bright_labeled = label(bright_mask)
                    
                    analysis_result['total_objects'] = int(np.max(labeled_sources))
                    analysis_result['bright_sources'] = int(np.max(bright_labeled))
                    analysis_result['faint_sources'] = analysis_result['total_objects'] - analysis_result['bright_sources']
                    
                    # Extract source properties for bright sources
                    if analysis_result['bright_sources'] > 0:
                        regions = regionprops(bright_labeled, intensity_image=img_sub)
                        
                        source_positions = []
                        photometry_data = []
                        
                        for region in regions[:20]:  # Limit to top 20 sources
                            # Source position (centroid)
                            y_center, x_center = region.centroid
                            source_positions.append([float(x_center), float(y_center)])
                            
                            # Photometric measurements
                            total_flux = float(region.intensity_image.sum())
                            peak_flux = float(region.max_intensity)
                            area = float(region.area)
                            
                            photometry_data.append({
                                'flux_total': total_flux,
                                'flux_peak': peak_flux,
                                'area_pixels': area,
                                'magnitude_estimate': -2.5 * np.log10(max(total_flux, 1e-10)) + 25.0,  # Rough magnitude
                                'snr': peak_flux / background_noise if background_noise > 0 else 0.0
                            })
                        
                        analysis_result['source_positions'] = source_positions
                        analysis_result['photometry_data'] = photometry_data
                
                # Calculate overall quality score
                quality_factors = []
                
                # Factor 1: Signal-to-noise ratio
                if background_noise > 0:
                    typical_snr = (max_val - background_level) / background_noise
                    quality_factors.append(min(typical_snr / 100.0, 1.0))
                
                # Factor 2: Saturation (lower is better)
                quality_factors.append(1.0 - min(analysis_result['saturation_fraction'] * 10, 1.0))
                
                # Factor 3: Number of detected sources (normalized)
                quality_factors.append(min(analysis_result['total_objects'] / 100.0, 1.0))
                
                # Factor 4: Dynamic range (normalized)
                if analysis_result['dynamic_range'] > 0:
                    quality_factors.append(min(analysis_result['dynamic_range'] / 60000.0, 1.0))
                
                analysis_result['quality_score'] = float(np.mean(quality_factors) if quality_factors else 0.0)
                
        except Exception as e:
            analysis_result['processing_status'] = f'ERROR_{str(e)[:50]}'
        
        results.append(analysis_result)
    
    return iter(results)

def extract_telescope_id(file_path):
    """Extract telescope identifier from file path"""
    # ZTF files typically have identifiers in the filename
    import re
    match = re.search(r'(\d{10})', file_path)
    return match.group(1)[:4] if match else 'UNKNOWN'

def extract_observation_time(file_path):
    """Extract observation timestamp from filename"""
    import re
    match = re.search(r'(\d{10})', file_path)
    if match:
        # Convert to a readable timestamp (simplified)
        obs_id = match.group(1)
        return f"2025-{obs_id[2:4]}-{obs_id[4:6]}T{obs_id[6:8]}:{obs_id[8:10]}:00"
    return "UNKNOWN"

def extract_image_type(file_path):
    """Determine image type from filename"""
    if 'difference' in file_path:
        return 'DIFFERENCE'
    elif 'science' in file_path:
        return 'SCIENCE'
    elif 'template' in file_path:
        return 'TEMPLATE'
    return 'UNKNOWN'

def main():
    print("Starting Advanced Astronomical Object Detection Pipeline...")
    
    # Input data path
    INPUT = "hdfs://namenode:8020/user/astro/raw/ztf_fits/*.fits"
    
    # Load FITS files with optimized partitioning
    fits_rdd = sc.binaryFiles(INPUT, minPartitions=4)
    
    print(f"Processing {fits_rdd.count()} FITS files...")
    
    # Apply advanced analysis
    analysis_rdd = fits_rdd.mapPartitions(advanced_fits_analysis)
    
    # Convert to DataFrame for SQL operations
    schema = StructType([
        StructField("file_path", StringType(), True),
        StructField("telescope_id", StringType(), True),
        StructField("observation_time", StringType(), True),
        StructField("image_type", StringType(), True),
        StructField("total_objects", IntegerType(), True),
        StructField("bright_sources", IntegerType(), True),
        StructField("faint_sources", IntegerType(), True),
        StructField("background_level", FloatType(), True),
        StructField("background_noise", FloatType(), True),
        StructField("dynamic_range", FloatType(), True),
        StructField("saturation_fraction", FloatType(), True),
        StructField("source_positions", StringType(), True),  # JSON string
        StructField("photometry_data", StringType(), True),   # JSON string
        StructField("quality_score", FloatType(), True),
        StructField("processing_status", StringType(), True)
    ])
    
    # Convert complex fields to JSON strings for DataFrame compatibility
    def serialize_complex_fields(record):
        record_dict = record
        record_dict['source_positions'] = json.dumps(record_dict['source_positions'])
        record_dict['photometry_data'] = json.dumps(record_dict['photometry_data'])
        return record_dict
    
    serialized_rdd = analysis_rdd.map(serialize_complex_fields)
    
    # Create DataFrame
    df = spark.createDataFrame(serialized_rdd, schema)
    
    # Cache for multiple operations
    df.cache()
    
    print("Generating Astronomical Analytics Report...")
    
    # Overall statistics
    total_files = df.count()
    successful_files = df.filter(col("processing_status") == "SUCCESS").count()
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total files processed: {total_files}")
    print(f"Successfully processed: {successful_files}")
    print(f"Success rate: {successful_files/total_files*100:.1f}%")
    
    # Object detection statistics by image type
    print(f"\n=== OBJECT DETECTION SUMMARY ===")
    detection_stats = df.filter(col("processing_status") == "SUCCESS") \
                       .groupBy("image_type") \
                       .agg(
                           count("*").alias("num_images"),
                           avg("total_objects").alias("avg_objects"),
                           avg("bright_sources").alias("avg_bright"),
                           avg("quality_score").alias("avg_quality")
                       ).collect()
    
    for row in detection_stats:
        print(f"{row['image_type']:12}: {row['num_images']:4} images, "
              f"avg objects: {row['avg_objects']:6.1f}, "
              f"avg bright: {row['avg_bright']:6.1f}, "
              f"avg quality: {row['avg_quality']:5.3f}")
    
    # Quality distribution
    print(f"\n=== QUALITY ASSESSMENT ===")
    quality_ranges = [
        ("Excellent", 0.8, 1.0),
        ("Good", 0.6, 0.8),
        ("Fair", 0.4, 0.6),
        ("Poor", 0.0, 0.4)
    ]
    
    for label, min_q, max_q in quality_ranges:
        count_in_range = df.filter(
            (col("quality_score") >= min_q) & 
            (col("quality_score") < max_q)
        ).count()
        print(f"{label:10}: {count_in_range:4} files ({count_in_range/total_files*100:5.1f}%)")
    
    # Save comprehensive results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = f"hdfs://namenode:8020/user/spark/gold/astronomical_analysis_{timestamp}"
    
    print(f"\nSaving results to: {output_path}")
    
    # Write as parquet for efficient querying
    df.coalesce(2).write.mode("overwrite").parquet(output_path)
    
    # Also save summary statistics as JSON
    summary_stats = {
        "processing_timestamp": timestamp,
        "total_files": total_files,
        "successful_files": successful_files,
        "success_rate": successful_files/total_files,
        "detection_stats": [row.asDict() for row in detection_stats],
        "quality_distribution": {
            label: df.filter((col("quality_score") >= min_q) & (col("quality_score") < max_q)).count()
            for label, min_q, max_q in quality_ranges
        }
    }
    
    summary_path = f"hdfs://namenode:8020/user/spark/gold/summary_astronomical_analysis_{timestamp}.json"
    sc.parallelize([json.dumps(summary_stats, indent=2)]).coalesce(1).saveAsTextFile(summary_path)
    
    print(f"Summary statistics saved to: {summary_path}")
    print("\nAdvanced Astronomical Analysis Pipeline completed successfully!")
    
    spark.stop()

if __name__ == "__main__":
    main()