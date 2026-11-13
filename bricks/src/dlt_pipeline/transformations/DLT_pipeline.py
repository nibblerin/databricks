from pyspark import pipelines as dp
from pyspark.sql import functions as F
 
VOLUME = "/Volumes/aaa/raw_files/data"
SOURCE_DRIVERS = "aaa.bronze.drivers"
CATALOG = "aaa_dp"
BRONZE  = f"{CATALOG}.bronze"
SILVER  = f"{CATALOG}.silver"
GOLD    = f"{CATALOG}.gold"
 
 
 #Bronze: Use streaming tables (@dlt.table) to read raw data for trips, drivers, and users.

@dp.table(name=f"{BRONZE}.users")
def bronze_users():
    # Auto Loader => streaming table (batch spark.read дал бы materialized view, из которого нельзя читать стримом)
    return (spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "avro")
            .option("pathGlobFilter", "*.avro")
            .load(VOLUME))
 
@dp.table(name=f"{BRONZE}.taxi", table_properties={"delta.feature.timestampNtz": "supported"})
def bronze_taxi():
    return (spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "parquet")
            .option("pathGlobFilter", "*.parquet")
            .load(VOLUME))
    
@dp.table(name=f"{BRONZE}.drivers")
def bronze_drivers():

    # skipChangeCommits in true is for RESTORE protection, but it would be applicable only after pipeline runs and RESTORING happens
    return (spark.readStream
            .option("skipChangeCommits", "true")
            .table(SOURCE_DRIVERS))

#Silver: Use Data Quality expectations (@dlt.expect_or_drop) to automatically filter out bad records (e.g., zero distance/fare, invalid ratings) and enrich the data with text statuses.

@dp.table(name=f"{SILVER}.taxi", table_properties={"delta.feature.timestampNtz": "supported"})
@dp.expect_or_drop("passenger_count_positive", "passenger_count > 0")
@dp.expect_or_drop("trip_distance_positive", "trip_distance > 0")
@dp.expect_or_drop("fare_amount_positive", "fare_amount > 0")
def silver_taxi():
    return (spark.readStream.table(f"{BRONZE}.taxi")
            .dropDuplicates()                                 
            .withColumn("description",
                        F.when(F.col("payment_type") == 1, "Credit Card")
                         .when(F.col("payment_type") == 2, "Cash")
                         .otherwise("Other"))
            .withColumn("ingest_time", F.current_timestamp()))

@dp.temporary_view(name="drivers_valid")
@dp.expect_or_drop("experience_positive", "experience > 0")
@dp.expect_or_drop("rating_in_range", "rating BETWEEN 1.0 AND 5.0")
def drivers_valid():
    return spark.readStream.table(f"{BRONZE}.drivers")
 
@dp.table(name=f"{SILVER}.drivers")
def silver_drivers():
    return (spark.readStream.table("drivers_valid")
            .withColumn("driver_id", F.sha2(F.concat_ws("|", "name", "car_number"), 256))
            .dropDuplicates(["driver_id"])
            .withColumn("ingest_time", F.current_timestamp()))


@dp.table(name=f"{SILVER}.users")
@dp.expect_or_drop("name_not_null", "name IS NOT NULL")
def silver_users():
    return (spark.readStream.table(f"{BRONZE}.users")
            .dropDuplicates()
            .withColumn("ingest_time", F.current_timestamp())
            .withColumn("registration_date",
                        F.expr("date_sub(current_date(), cast(rand() * 365 as int))")))

#Gold: Calculate the aggregates (trips and revenue per zone) and join the trip data with driver profiles.
 
@dp.materialized_view(name=f"{GOLD}.taxi_metrics", cluster_by=["PULocationID", "description"])
def gold_taxi_metrics():
    return (spark.read.table(f"{SILVER}.taxi")
            .groupBy("PULocationID", "description")
            .agg(F.count("*").alias("total_trips"),
                 F.round(F.sum("fare_amount"), 2).alias("total_revenue")))
    
@dp.materialized_view(name=f"{GOLD}.user_metrics")
def gold_user_metrics():
    return (spark.read.table(f"{SILVER}.users")
            .groupBy("registration_date")
            .agg(F.count("*").alias("user_count"))
            .orderBy(F.desc("user_count")))

#Databricks recommends liquid clustering instead of Z-ordering for optimizing data layout in pipeline tables.
@dp.materialized_view(name=f"{GOLD}.driver_metrics", table_properties={"pipelines.autoOptimize.zOrderCols": "experience"})
def gold_driver_metrics():
    return (spark.read.table(f"{SILVER}.drivers")
            .groupBy("experience")
            .agg(F.countDistinct("driver_id").alias("drivers_count"),
                 F.round(F.avg("rating"), 2).alias("avg_rating")))

# the concept for separation below came from the databricks blog: If having a non-deterministic function is necessary for your analysis (my urge forskewness makes it necessary), a better approach is to push that value into the source table itself, rather than calculating it dynamically in the materialized view. .. accomplish this by moving the random_number column to pull from the source table instead of adding it in at the MV level.

@dp.table(name=f"{SILVER}.taxi_assigned_drivers", table_properties={"delta.feature.timestampNtz": "supported"})
def silver_taxi_assigned_drivers():
    ids_df = (spark.read.table(f"{SILVER}.drivers")
              .select("driver_id").distinct()
              .agg(F.sort_array(F.collect_list("driver_id")).alias("ids"))
              .withColumn("n", F.size("ids")))

    return (spark.readStream.table(f"{SILVER}.taxi")
            .crossJoin(F.broadcast(ids_df))
            .withColumn("driver_id", F.col("ids")[F.floor(F.pow(F.rand(42), 2) * F.col("n")).cast("int")])
            .drop("ids", "n"))
    
@dp.materialized_view(name=f"{GOLD}.enriched_trips", table_properties={"delta.feature.timestampNtz": "supported"})
def gold_enriched_trips():
    taxi = spark.read.table(f"{SILVER}.taxi_assigned_drivers")
    drivers = spark.read.table(f"{SILVER}.drivers").withColumnRenamed("ingest_time", "driver_ingest_time")
    return taxi.join(drivers, on="driver_id", how="left")