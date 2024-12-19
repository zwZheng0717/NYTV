from flask import Flask, render_template, request, send_file, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, sum as spark_sum, avg
import folium
from folium.plugins import HeatMap
from folium import TileLayer
import pandas as pd
import os
import requests

# 初始化 Flask 应用
app = Flask(__name__, static_folder="static")

# Google 瓦片 URL
GOOGLE_TILE_URL = "https://mt.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"

# 默认 HDFS 数据文件路径
HDFS_DATA_PATH = "hdfs://192.168.1.9:8020/data/data.csv"  # HDFS Namenode 地址
hdfs_temp_output_path = "hdfs://192.168.1.9:8020/output/temp_processed_data"  # HDFS 临时输出路径
hdfs_final_output_path = "hdfs://192.168.1.9:8020/output/processed_data.csv"  # HDFS 最终输出路径

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("NYC Traffic Heatmap") \
    .master("spark://192.168.1.9:7077") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "20") \
    .config("spark.driver.memory", "2g") \
    .config("spark.driver.cores", "2") \
    .config("spark.driver.bindAddress", "192.168.1.9") \
    .config("spark.driver.host", "192.168.1.9") \
    .getOrCreate()

# 加载 CSV 数据
raw_data = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(HDFS_DATA_PATH)

# 按经纬度、时间和街道信息聚合流量数据
aggregated_data = raw_data.groupBy(
    "lat", "lon", "year", "month", "day", "hour", "street"
).agg(
    spark_sum("Vol").alias("total_volume"),
    avg("Vol").alias("avg_volume")
)

aggregated_data = aggregated_data.coalesce(1)

# 将结果保存到 HDFS 临时目录
aggregated_data.write.format("csv") \
    .option("header", "true") \
    .mode("overwrite") \
    .save(hdfs_temp_output_path)

# 使用 WebHDFS API 获取生成的文件名
namenode_host = "192.168.1.9"
webhdfs_url = f"http://{namenode_host}:9870/webhdfs/v1/output/temp_processed_data/?op=LISTSTATUS"
response = requests.get(webhdfs_url)
response.raise_for_status()
file_status = response.json()
files = file_status['FileStatuses']['FileStatus']
csv_file_name = [file['pathSuffix'] for file in files if file['pathSuffix'].startswith('part')][0]
print("success")

# 使用 WebHDFS API 重命名文件
rename_url = f"http://{namenode_host}:9870/webhdfs/v1/output/temp_processed_data/{csv_file_name}?op=RENAME&destination=/output/processed_data.csv&user.name=root"
rename_response = requests.put(rename_url)
rename_response.raise_for_status()

print(f"Processed data saved to {hdfs_final_output_path}")

use_hdfs = True

@app.route("/", methods=["GET", "POST"])
def index():
    # 加载 CSV 数据
    aggregated_data = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(hdfs_final_output_path)
        
    if not use_hdfs:
        aggregated_data = pd.read_csv("predict_month.csv")
        aggregated_data = spark.createDataFrame(aggregated_data)

    # 获取基础信息（年份、月份、小时的范围）
    min_year = int(aggregated_data.select("year").rdd.min()[0])
    max_year = int(aggregated_data.select("year").rdd.max()[0])
    min_month, max_month = 1, 12
    min_hour, max_hour = 0, 23
    
    # 默认选择范围
    selected_start_year = min_year
    selected_end_year = max_year
    selected_start_month = min_month
    selected_end_month = max_month
    selected_start_hour = min_hour
    selected_end_hour = max_hour

    filtered_data = aggregated_data
    
    pandas_data = filtered_data.toPandas()

    if request.method == "POST":
        # 获取用户选择的时间范围
        selected_start_year = int(request.form.get("start_year", min_year))
        selected_end_year = int(request.form.get("end_year", max_year))
        selected_start_month = int(request.form.get("start_month", min_month))
        selected_end_month = int(request.form.get("end_month", max_month))
        selected_start_hour = int(request.form.get("start_hour", min_hour))
        selected_end_hour = int(request.form.get("end_hour", max_hour))

        # 过滤数据（利用 Spark DataFrame）
        filtered_data = aggregated_data.filter(
            (col("year") >= selected_start_year) & (col("year") <= selected_end_year) &
            (col("month") >= selected_start_month) & (col("month") <= selected_end_month) &
            (col("hour") >= selected_start_hour) & (col("hour") <= selected_end_hour)
        )

        # 将数据转换为 Pandas DataFrame，并生成热力图
        pandas_data = filtered_data.toPandas()
        print(pandas_data)
        heat_data = pandas_data[["lat", "lon", "total_volume"]].values.tolist()

        # 创建热力图
        nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12, tiles=None)
        TileLayer(tiles=GOOGLE_TILE_URL, attr="© Google Maps", name="Google Maps", overlay=False).add_to(nyc_map)
        HeatMap(heat_data, min_opacity=0.5, max_val=max(pandas_data["total_volume"])).add_to(nyc_map)
        nyc_map.save("./static/map.html")

    return render_template(
        "index.html",
        min_year=min_year,
        max_year=max_year,
        selected_start_year=selected_start_year,
        selected_end_year=selected_end_year,
        min_month=min_month,
        max_month=max_month,
        selected_start_month=selected_start_month,
        selected_end_month=selected_end_month,
        min_hour=min_hour,
        max_hour=max_hour,
        selected_start_hour=selected_start_hour,
        selected_end_hour=selected_end_hour,
        filtered_data=pandas_data.to_html()  # 传递数据到前端页面
    )
    
@app.route("/switch_data_file", methods=["POST"])
def switch_data_file():
    global use_hdfs
    new_data_file = request.json.get("data_file")
    
    if new_data_file == "processed_data_with_year.csv":
        use_hdfs = True  # 切换到 HDFS 数据
    elif new_data_file == "predict_month.csv":
        use_hdfs = False  # 切换到本地预测数据

    return jsonify({"status": "success", "data_file": new_data_file})

@app.route("/download_hot_streets", methods=["GET"])
def download_hot_streets():
    try:
        # 从本地文件读取过滤后的数据
        hot_streets_path = "./static/hot_streets.csv"

        return send_file(hot_streets_path, as_attachment=True)

    except Exception as e:
        print(f"Error processing hot streets: {e}")
        return jsonify({"status": "error", "message": "Failed to process hot streets."})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
