package org.heatmap;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.*;
import org.apache.spark.sql.functions;
import org.apache.sedona.sql.utils.SedonaSQLRegistrator;

public class NYCHeatMap {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("NYC Traffic Heatmap").setMaster("local[*]");
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        SedonaSQLRegistrator.registerAll(spark);

        // 加载数据
        Dataset<Row> rawData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/data.csv");

        Dataset<Row> withGeometry = rawData.withColumn("geometry",
                functions.expr("ST_Point(CAST(ST_X(ST_GeomFromWKT(WktGeom)) AS DOUBLE), " +
                        "CAST(ST_Y(ST_GeomFromWKT(WktGeom)) AS DOUBLE))"));

        // 转换坐标系：从 EPSG:2263 (NY State Plane) 到 EPSG:4326 (WGS 84)
        Dataset<Row> withLatLon = withGeometry.withColumn("geometry_transformed",
                        functions.expr("ST_Transform(geometry, 'EPSG:2263', 'EPSG:4326')"))
                .withColumn("lat", functions.expr("ST_X(geometry_transformed)"))
                .withColumn("lon", functions.expr("ST_Y(geometry_transformed)"));

        // 按经纬度、年、月、日、小时、WktGeom、street、fromSt 和 toSt 聚合流量数据
        Dataset<Row> aggregatedData = withLatLon.groupBy(
                        "lat",
                        "lon",
                        "Yr",
                        "M",
                        "D",
                        "HH",
                        "street")
                .agg(functions.sum("Vol").alias("total_volume"),
                        functions.avg("Vol").alias("avg_volume"));

        Dataset<Row> result = aggregatedData
                .withColumnRenamed("Yr", "year")
                .withColumnRenamed("M", "month")
                .withColumnRenamed("D", "day")
                .withColumnRenamed("HH", "hour");

        result.show();

        // 合并所有分区为一个分区
        Dataset<Row> singlePartitionResult = result.coalesce(1);

        singlePartitionResult.write()
                .format("csv")
                .option("header", "true")
                .mode("overwrite")
                .save("processed_traffic_data");

        spark.stop();
    }
}
