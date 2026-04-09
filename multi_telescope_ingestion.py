# app/multi_telescope_ingestion.py
"""
Multi-Telescope Data Ingestion Pipeline
Handles data from multiple telescope sources with adaptive processing
"""

import os
import re
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, regexp_extract, size, split
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

def create_spark_session():
    return (
        SparkSession.builder
        .appName("Multi-Telescope-Ingestion")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        .config("spark.executor.memory", "1g")
        .config("spark.executor.cores", "1")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

class TelescopeDataProcessor:
    """Base class for telescope-specific data processing"""
    
    def __init__(self, telescope_name):
        self.telescope_name = telescope_name
        self.file_patterns = {}
        self.metadata_extractors = {}
    
    def register_file_pattern(self, data_type, pattern):
        """Register filename pattern for specific data type"""
        self.file_patterns[data_type] = pattern
    
    def register_metadata_extractor(self, field, extractor_func):
        """Register metadata extraction function"""
        self.metadata_extractors[field] = extractor_func
    
    def classify_file(self, file_path):
        """Classify file based on registered patterns"""
        filename = os.path.basename(file_path)
        for data_type, pattern in self.file_patterns.items():
            if re.search(pattern, filename):
                return data_type
        return "UNKNOWN"
    
    def extract_metadata(self, file_path):
        """Extract metadata using registered extractors"""
        metadata = {"telescope": self.telescope_name}
        for field, extractor in self.metadata_extractors.items():
            try:
                metadata[field] = extractor(file_path)
            except:
                metadata[field] = None
        return metadata

class ZTFProcessor(TelescopeDataProcessor):
    """Zwicky Transient Facility data processor"""
    
    def __init__(self):
        super().__init__("ZTF")
        
        # Register ZTF file patterns
        self.register_file_pattern("SCIENCE", r"_science\.fits$")
        self.register_file_pattern("DIFFERENCE", r"_difference\.fits$")
        self.register_file_pattern("TEMPLATE", r"_template\.fits$")
        
        # Register metadata extractors
        self.register_metadata_extractor("field_id", self._extract_field_id)
        self.register_metadata_extractor("ccd_id", self._extract_ccd_id)
        self.register_metadata_extractor("filter_band", self._extract_filter)
        self.register_metadata_extractor("observation_date", self._extract_obs_date)
        self.register_metadata_extractor("exposure_id", self._extract_exposure_id)
    
    def _extract_field_id(self, file_path):
        """Extract ZTF field ID from filename"""
        match = re.search(r'(\d{6})', file_path)
        return int(match.group(1)[:3]) if match else None
    
    def _extract_ccd_id(self, file_path):
        """Extract CCD quadrant ID"""
        match = re.search(r'(\d{6})', file_path)
        return int(match.group(1)[3:5]) if match else None
    
    def _extract_filter(self, file_path):
        """Extract filter band (assumed from CCD ID pattern)"""
        # ZTF uses g, r, i bands - simplified extraction
        ccd_id = self._extract_ccd_id(file_path)
        if ccd_id:
            return ["g", "r", "i"][ccd_id % 3]
        return None
    
    def _extract_obs_date(self, file_path):
        """Extract observation date from filename"""
        match = re.search(r'(\d{10})', file_path)
        if match:
            date_str = match.group(1)
            # Convert to ISO format (simplified)
            year = "20" + date_str[:2]
            month = date_str[2:4]
            day = date_str[4:6]
            return f"{year}-{month}-{day}"
        return None
    
    def _extract_exposure_id(self, file_path):
        """Extract unique exposure identifier"""
        match = re.search(r'(\d{10})', file_path)
        return int(match.group(1)) if match else None

class HSCProcessor(TelescopeDataProcessor):
    """Hyper Suprime-Cam data processor (simulated)"""
    
    def __init__(self):
        super().__init__("HSC")
        
        # Simulated HSC patterns
        self.register_file_pattern("CALIBRATED", r"HSCA.*\.fits$")
        self.register_file_pattern("RAW", r"HSCR.*\.fits$")
        
        self.register_metadata_extractor("visit_id", self._extract_visit_id)
        self.register_metadata_extractor("detector_id", self._extract_detector_id)

    def _extract_visit_id(self, file_path):
        return hash(file_path) % 100000  # Simulated
    
    def _extract_detector_id(self, file_path):
        return hash(file_path) % 104  # HSC has 104 detectors

def process_telescope_data(spark, telescope_processor, input_paths):
    """Process data for a specific telescope"""
    
    print(f"Processing {telescope_processor.telescope_name} data...")
    
    # Collect all files
    all_files = []
    for path in input_paths:
        try:
            files_rdd = spark.sparkContext.wholeTextFiles(path)
            file_paths = files_rdd.keys().collect()
            all_files.extend(file_paths)
        except:
            print(f"Warning: Could not access path {path}")
    
    if not all_files:
        print(f"No files found for {telescope_processor.telescope_name}")
        return None
    
    print(f"Found {len(all_files)} files")
    
    # Process file metadata
    def extract_file_metadata(file_path):
        metadata = telescope_processor.extract_metadata(file_path)
        metadata['file_path'] = file_path
        metadata['file_size_mb'] = 0.02  # Placeholder
        metadata['data_type'] = telescope_processor.classify_file(file_path)
        metadata['processing_timestamp'] = datetime.utcnow().isoformat()
        return metadata
    
    # Create RDD of metadata
    files_rdd = spark.sparkContext.parallelize(all_files)
    metadata_rdd = files_rdd.map(extract_file_metadata)
    
    # Convert to DataFrame
    metadata_df = spark.createDataFrame(metadata_rdd)
    
    return metadata_df

def create_unified_catalog(spark, telescope_dataframes):
    """Create unified catalog from multiple telescope sources"""
    
    print("Creating unified astronomical data catalog...")
    
    # Define unified schema
    unified_schema = StructType([
        StructField("file_path", StringType(), False),
        StructField("telescope", StringType(), False),
        StructField("data_type", StringType(), True),
        StructField("observation_date", StringType(), True),
        StructField("field_id", IntegerType(), True),
        StructField("detector_id", IntegerType(), True),
        StructField("filter_band", StringType(), True),
        StructField("exposure_id", IntegerType(), True),
        StructField("file_size_mb", FloatType(), True),
        StructField("processing_timestamp", StringType(), True),
        StructField("quality_flags", StringType(), True)
    ])
    
    unified_dfs = []
    
    for telescope_name, df in telescope_dataframes.items():
        print(f"Standardizing {telescope_name} data...")
        
        # Add missing columns with defaults
        for field in unified_schema.fields:
            if field.name not in df.columns:
                df = df.withColumn(field.name, lit(None).cast(field.dataType))
        
        # Select only unified schema columns
        standardized_df = df.select(*[field.name for field in unified_schema.fields])
        unified_dfs.append(standardized_df)
    
    # Union all telescope data
    if unified_dfs:
        catalog_df = unified_dfs[0]
        for df in unified_dfs[1:]:
            catalog_df = catalog_df.union(df)
    else:
        catalog_df = spark.createDataFrame([], unified_schema)
    
    return catalog_df

def main():
    spark = create_spark_session()
    
    print("=== Multi-Telescope Data Ingestion Pipeline ===")
    print(f"Started at: {datetime.utcnow().isoformat()}")
    
    # Initialize telescope processors
    ztf_processor = ZTFProcessor()
    hsc_processor = HSCProcessor()  # Simulated
    
    # Process data from each telescope
    telescope_data = {}
    
    # ZTF data processing
    ztf_paths = ["hdfs://namenode:8020/user/astro/raw/ztf_fits/*.fits"]
    ztf_df = process_telescope_data(spark, ztf_processor, ztf_paths)
    if ztf_df:
        telescope_data["ZTF"] = ztf_df
    
    # HSC data processing (simulated - would be similar for real HSC data)
    # hsc_paths = ["hdfs://namenode:8020/user/astro/raw/hsc_fits/*.fits"]
    # hsc_df = process_telescope_data(spark, hsc_processor, hsc_paths)
    # if hsc_df:
    #     telescope_data["HSC"] = hsc_df
    
    if not telescope_data:
        print("No telescope data found. Exiting.")
        spark.stop()
        return
    
    # Create unified catalog
    unified_catalog = create_unified_catalog(spark, telescope_data)
    
    # Cache for analysis
    unified_catalog.cache()
    
    # Generate ingestion report
    print(f"\n=== INGESTION SUMMARY ===")
    total_files = unified_catalog.count()
    print(f"Total files cataloged: {total_files}")
    
    # Files by telescope
    telescope_counts = unified_catalog.groupBy("telescope").count().collect()
    for row in telescope_counts:
        print(f"  {row['telescope']}: {row['count']} files")
    
    # Files by data type
    print(f"\n=== DATA TYPE DISTRIBUTION ===")
    type_counts = unified_catalog.groupBy("telescope", "data_type").count().collect()
    for row in type_counts:
        print(f"  {row['telescope']} {row['data_type']}: {row['count']} files")
    
    # Save unified catalog
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    catalog_path = f"hdfs://namenode:8020/user/spark/silver/unified_catalog_{timestamp}"
    
    print(f"\nSaving unified catalog to: {catalog_path}")
    unified_catalog.coalesce(1).write.mode("overwrite").parquet(catalog_path)
    
    # Create data quality assessment
    print(f"\n=== DATA QUALITY ASSESSMENT ===")
    
    # Check for missing metadata
    missing_metadata = unified_catalog.select(
        *[col(field.name).isNull().alias(f"missing_{field.name}") 
          for field in unified_catalog.schema.fields if field.name not in ["quality_flags"]]
    )
    
    quality_summary = {}
    for field in missing_metadata.columns:
        missing_count = missing_metadata.filter(col(field) == True).count()
        quality_summary[field] = {
            "missing_count": missing_count,
            "missing_percentage": (missing_count / total_files * 100) if total_files > 0 else 0
        }
        print(f"  {field}: {missing_count} missing ({missing_count/total_files*100:.1f}%)")
    
    # Save quality report
    quality_path = f"hdfs://namenode:8020/user/spark/silver/quality_report_{timestamp}.json"
    quality_report = {
        "timestamp": timestamp,
        "total_files": total_files,
        "telescope_distribution": {row['telescope']: row['count'] for row in telescope_counts},
        "quality_metrics": quality_summary
    }
    
    import json
    spark.sparkContext.parallelize([json.dumps(quality_report, indent=2)]).coalesce(1).saveAsTextFile(quality_path)
    
    print(f"Quality report saved to: {quality_path}")
    print(f"\nMulti-Telescope Ingestion Pipeline completed at: {datetime.utcnow().isoformat()}")
    
    spark.stop()

if __name__ == "__main__":
    main()