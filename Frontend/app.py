from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import folium
from folium.plugins import HeatMap
from folium import TileLayer

# 加载数据
# data_file = "processed_data_with_year.csv"
data_file = "processed_data_with_year.csv"
data = pd.read_csv(data_file)

# Google 瓦片 URL
GOOGLE_TILE_URL = "https://mt.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"

app = Flask(__name__, static_folder="static")

@app.route("/", methods=["GET", "POST"])
def index():
    # 初始化年份选择范围
    min_year = int(data["year"].min())
    max_year = int(data["year"].max())
    selected_start_year = min_year
    selected_end_year = max_year

    min_month = 1
    max_month = 12
    selected_start_month = min_month
    selected_end_month = max_month

    # 初始化时间选择范围
    min_hour = 0
    max_hour = 23
    selected_start_hour = min_hour
    selected_end_hour = max_hour

    filtered_data = data

    if request.method == "POST":
        # 获取用户选择的年份区间
        selected_start_year = int(request.form.get("start_year", min_year))
        selected_end_year = int(request.form.get("end_year", max_year))

        # 获取用户选择的时间段
        selected_start_month = int(request.form.get("start_month", min_month))
        selected_end_month = int(request.form.get("end_month", max_month))

        # 获取用户选择的时间段
        selected_start_hour = int(request.form.get("start_hour", min_hour))
        selected_end_hour = int(request.form.get("end_hour", max_hour))

        # 过滤数据
        filtered_data = data[
            (data["year"] >= selected_start_year) &
            (data["year"] <= selected_end_year) &
            (data["month"] >= selected_start_month) &
            (data["month"] <= selected_end_month) &
            (data["hour"] >= selected_start_hour) &
            (data["hour"] <= selected_end_hour)
        ]

        # 创建地图
        nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12, tiles=None)

        # 添加 Google 瓦片层
        TileLayer(
            tiles=GOOGLE_TILE_URL,
            attr="© Google Maps",
            name="Google Maps",
            overlay=False
        ).add_to(nyc_map)

        # 添加热力图
        heat_data = filtered_data[["lat", "lon", "avg_volume"]].values.tolist()
        HeatMap(heat_data, min_opacity=0.5, max_val=filtered_data["avg_volume"].max()).add_to(nyc_map)

        # 保存地图到文件
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
        selected_end_hour=selected_end_hour
    )

@app.route("/switch_data_file", methods=["POST"])
def switch_data_file():
    global data_file, data
    # 从请求体获取新的数据文件
    new_data_file = request.json.get("data_file")
    
    if new_data_file:
        # 更新数据文件路径
        data_file = new_data_file
        data = pd.read_csv(data_file)
    
    # 返回确认信息
    return jsonify({"status": "success", "data_file": data_file})

@app.route("/download_hot_streets", methods=["GET"])
def download_hot_streets():
    # 统计前 50 个最热点街道
    hot_streets = data.groupby("street").agg({"total_volume": "sum"}).reset_index()
    hot_streets = hot_streets.sort_values("total_volume", ascending=False).head(50)

    # 保存为 CSV 文件
    hot_streets_file = "./static/hot_streets.csv"
    hot_streets.to_csv(hot_streets_file, index=False)

    # 提供下载
    return send_file(hot_streets_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
