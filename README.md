# A Scalable Big Data Analytics Framework for Time-Domain Astronomy

## Overview
[cite_start]This project delivers a horizontally scalable big data analytics platform built on **Apache Spark 3.5.7** and **HDFS**[cite: 38, 128]. [cite_start]It is designed to handle the complex ingestion and analysis of **122,382 FITS files** from the Zwicky Transient Facility (ZTF)[cite: 29, 30, 129]. [cite_start]The framework successfully validates the scientific principle of image subtraction to isolate transient objects while maintaining strong data governance through the **Medallion Architecture**[cite: 39, 131, 132].

---

## Technical Stack
* [cite_start]**Cluster Manager**: Apache Spark 3.5.7 (Containerized)[cite: 38].
* [cite_start]**Storage**: Hadoop Distributed File System (HDFS)[cite: 38].
* [cite_start]**Data Governance**: Medallion Architecture (Bronze, Silver, Gold layers)[cite: 39].
* [cite_start]**Scientific Libraries**: SciPy, Scikit-learn, and Astropy[cite: 40].
* [cite_start]**Data Formats**: FITS (Raw), Parquet (Structured), and JSON (Reports)[cite: 18, 58, 123].

---

## Pipeline Architecture
[cite_start]The project is partitioned into four modular automated pipelines[cite: 63]:

### 1. Bronze Pipeline (`01_bronze_pipeline.py`)
* [cite_start]Performs raw binary ingestion via Spark's RDD API[cite: 46, 65].
* [cite_start]Executes astronomical feature extraction tasks like background normalization using median intensity[cite: 48, 49, 50].
* [cite_start]Implements **3-sigma thresholding** to flag bright sources[cite: 53, 67].

### 2. Silver Pipeline (`02_silver_pipeline.py`)
* [cite_start]Focuses on data restructuring and filtering for successful extractions[cite: 68].
* [cite_start]Utilizes a **Pivot Operation** to collapse Science, Template, and Difference rows into a single wide-format record, reducing row count by **66.7%**[cite: 68, 124].

### 3. Gold Layer & Clustering (`03_gold_clustering.py`)
* [cite_start]Normalizes features using `VectorAssembler` and `StandardScaler`[cite: 71].
* [cite_start]Trains a **K-Means algorithm ($k=3$)** to categorize objects based on transient characteristics[cite: 72].

### 4. Final Feature Enrichment (`04_gold_final_features.py`)
* [cite_start]Adds spatial (Right Ascension and Declination) and temporal dimensions[cite: 73, 74].
* [cite_start]Uses User-Defined Functions (UDFs) to parse ZTF object IDs for observation timestamps[cite: 74].

---

## Performance Tuning
[cite_start]Initial configurations faced "no tasks running" failure modes due to excessive task overhead[cite: 80]. The following optimizations were implemented:
* [cite_start]**Partition Optimization**: Reduced Spark partitions from **512** to **4**, resulting in a **95% reduction** in task scheduling overhead[cite: 81, 103].
* [cite_start]**Memory Management**: Configured executor memory to **1 GB** to prevent Out-of-Memory errors during image array manipulations[cite: 82, 103].
* [cite_start]**Throughput**: Achieved stable execution, processing the entire dataset in **1,098 seconds (~18 minutes)**[cite: 87, 103].

---

## Key Results
| Metric | Result |
| :--- | :--- |
| **Total Files Processed** | [cite_start]122,382 FITS files (2.29 GB) [cite: 30, 103] |
| **K-Means Silhouette Score** | [cite_start]0.6842 (Strong structural separation) [cite: 119] |
| **Avg Objects (Template)** | [cite_start]6.35 [cite: 108] |
| **Avg Objects (Difference)** | [cite_start]2.79 [cite: 108] |
| **Quality Distribution** | [cite_start]95.8% classified as Fair or Poor [cite: 61, 115] |

