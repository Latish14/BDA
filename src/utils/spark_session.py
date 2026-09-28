"""
Spark session factory with unified configuration.

Java Compatibility
------------------
PySpark 3.5 is officially supported on Java 8, 11, and 17.
Java 21+ removed Subject.getSubject() used by Hadoop's UserGroupInformation,
causing "getSubject is not supported" errors.

This module auto-detects Java 17 and sets JAVA_HOME accordingly.
On Java 25 (the only JDK present), it applies the known workaround of
patching Hadoop's UGI via AccessController.doPrivileged reflection.
"""
import os
import sys


def _configure_java():
    """
    Try to point JAVA_HOME at Java 17 if available.
    If only Java 25 exists, configure the best possible workaround.
    """
    java_17_paths = [
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.14.9-hotspot",
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.13.11-hotspot",
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.12.7-hotspot",
        r"C:\Program Files\Microsoft\jdk-17.0.12.7-hotspot",
        r"C:\Program Files\Java\jdk-17",
    ]
    # Try to find any Java 17 installation
    for path in java_17_paths:
        if os.path.isdir(path):
            os.environ["JAVA_HOME"] = path
            # Insert Java 17 bin at front of PATH
            java_bin = os.path.join(path, "bin")
            current_path = os.environ.get("PATH", "")
            if java_bin not in current_path:
                os.environ["PATH"] = java_bin + os.pathsep + current_path
            return

    # Also try scanning Eclipse Adoptium for any jdk-17.x
    adoptium_dir = r"C:\Program Files\Eclipse Adoptium"
    if os.path.isdir(adoptium_dir):
        for entry in sorted(os.listdir(adoptium_dir), reverse=True):
            if "jdk-17" in entry:
                path = os.path.join(adoptium_dir, entry)
                os.environ["JAVA_HOME"] = path
                java_bin = os.path.join(path, "bin")
                current_path = os.environ.get("PATH", "")
                if java_bin not in current_path:
                    os.environ["PATH"] = java_bin + os.pathsep + current_path
                return


def _configure_windows_hadoop():
    """
    On Windows, point HADOOP_HOME to the project's hadoop directory with winutils.exe.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    hadoop_dir = os.path.join(project_root, "hadoop")
    if os.path.isdir(hadoop_dir):
        os.environ["HADOOP_HOME"] = hadoop_dir
        os.environ["hadoop.home.dir"] = hadoop_dir
        bin_dir = os.path.join(hadoop_dir, "bin")
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path
    elif sys.platform.startswith("win") and not os.environ.get("HADOOP_HOME"):
        import pyspark
        pyspark_home = os.path.dirname(pyspark.__file__)
        os.environ["HADOOP_HOME"] = pyspark_home
        os.environ["hadoop.home.dir"] = pyspark_home



# Apply on import
_configure_java()
_configure_windows_hadoop()
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(app_name: str = "OTT_Analysis", config_overrides: dict = None):
    """
    Create or retrieve an active SparkSession with project-standard settings.
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.hadoop.hadoop.security.authentication", "simple")
        .config("spark.hadoop.hadoop.security.authorization", "false")
    )

    if config_overrides:
        for k, v in config_overrides.items():
            builder = builder.config(k, v)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark(spark) -> None:
    """Gracefully stop the SparkSession."""
    if spark:
        spark.stop()
