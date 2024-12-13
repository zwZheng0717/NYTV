<img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg" alt="Apache Spark Icon" width="600">

# :dizzy:Spark处理纽约市交通流量信息-预测与热力地图可视化

## 项目简介

本项目旨在处理纽约市交通数据，通过 Apache Spark 进行大规模数据处理、使用深度学习模型进行交通预测，并通过 Web 界面进行可视化展示。项目分为三个主要模块：

- **预测（Predict）**：基于 D2STGNN（Decoupled Dynamic Spatial-Temporal Graph Neural Network）模型进行交通流量预测。
- **数据处理（DataProcess）**：使用 Apache Spark 和 Apache Sedona 对原始交通数据进行清洗、预处理，并构建交通路网图。
- **前端展示（Frontend）**：通过 Web 界面展示交通流量数据和预测结果，用户可以查看热力图及切换实际与预测数据。


## 项目结构

```
NYTV-main/
├── DataProcess/
│   ├── pom.xml
│   ├── processed_traffic_data/
│   │   └── ...
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/
│   │   │   │   └── org/
│   │   │   │       └── heatmap/
│   │   │   │           └── NYCHeatMap.java
│   │   │   └── resources/
│   │   │       └── data.csv
│   │   └── ...
│   └── .gitignore
│   └── .idea/
│       ├── .gitignore
│       ├── encodings.xml
│       ├── misc.xml
│       └── ...
├── Frontend/
│   ├── app.py
│   ├── processed_data_with_hour.csv
│   ├── processed_data_with_year.csv
│   ├── static/
│   │   ├── hot_streets.csv
│   │   └── map.html
│   ├── templates/
│   │   ├── index.html
│   │   └── map.html
│   └── ...
├── Predict/
│   ├── configs/
│   │   ├── NYTV.yaml
│   │   └── ...
│   ├── datasets/
│   │   ├── raw_data/
│   │   │   └── NYTV/
│   │   │       ├── generate_training_data.py
│   │   │       ├── generate_adj_mx.py
│   │   │       └── ...
│   │   └── ...
│   ├── main.py
│   ├── models/
│   │   └── ...
│   ├── requirements.txt
│   └── ...
├── README.md
└── ...
```

## 环境依赖

### 数据处理模块

- **Java 1.8**
- **Apache Maven**
- **Apache Spark 3.4.0**
- **Apache Sedona 1.3.0-incubating**

### 预测模块

- **Python 3.10**
- **PyTorch**
- **依赖库**（详见 `requirements.txt`）

### 前端展示模块

- **Flask**
- **Pandas**
- **Folium**

## 数据集

本项目使用了纽约市交通局（NYC DOT）的[自动交通流量计数数据集](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/about_data)。该数据集包含纽约市桥梁和道路的交通流量信息，由自动交通记录仪（ATR）采集。

- **数据更新时间**：2024 年 9 月 3 日
- **数据范围**：桥梁和道路上的交通流量数据，包含车辆计数、日期、时间、位置、方向等信息。
- **数据量**：约 171 万行，14 列
- **数据字段**：
  - `RequestID`：每个计数请求的唯一 ID
  - `Boro`：所在行政区
  - `Yr`、`M`、`D`：计数日期（年、月、日）
  - `HH`、`MM`：计数时间（小时、分钟）
  - `Vol`：15 分钟内的车辆计数总和
  - `SegmentID`：街道段 ID
  - `WktGeom`：空间坐标信息（Well-Known Text 格式）
  - `street`、`fromSt`、`toSt`：所在街道、起点街道、终点街道
  - `Direction`：交通方向

## 预测模型D2STGNN

预测模块使用了 **D2STGNN（Decoupled Dynamic Spatial-Temporal Graph Neural Network）** 模型，这是一个用于交通预测的先进深度学习模型。该模型能够有效地建模交通数据中的复杂时空相关性。

![D2STGNN](https://github.com/GestaltCogTeam/D2STGNN/raw/github/figures/D2STGNN.png)

- **VLDB'22 paper: **["Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting"](https://arxiv.org/abs/2202.04179)

- **模型特点**：
  - **解耦**：将交通数据的扩散信号和固有信号进行解耦，分别处理，提高预测性能。
  - **动态图学习**：通过动态图学习模块，捕获交通网络的动态特性。
  - **高性能**：在多个真实交通数据集上取得了先进的预测效果。
  



---



## 数据收集与处理

在` NYCHeatMap.java `中,利用 Spark 对大规模交通数据进行高效处理，包括数据清洗、格式转换和特征提取。

### 1. 数据读取与坐标转换

```java
// 加载数据
Dataset<Row> rawData = spark.read()
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("src/main/resources/data.csv");

// 注册 Sedona SQL 函数
SedonaSQLRegistrator.registerAll(spark);

// 创建几何列并转换坐标系
Dataset<Row> withGeometry = rawData.withColumn("geometry",
    functions.expr("ST_GeomFromWKT(WktGeom)"));
Dataset<Row> withLatLon = withGeometry.withColumn("geometry_transformed",
        functions.expr("ST_Transform(geometry, 'EPSG:2263', 'EPSG:4326')"))
    .withColumn("lat", functions.expr("ST_Y(geometry_transformed)"))
    .withColumn("lon", functions.expr("ST_X(geometry_transformed)"));
```

### 2. 数据聚合

```java
// 按经纬度、时间和街道名称聚合流量数据
Dataset<Row> aggregatedData = withLatLon.groupBy(
        "lat",
        "lon",
        "Yr",
        "M",
        "D",
        "HH",
        "street",
        "fromSt",
        "toSt")
    .agg(functions.sum("Vol").alias("total_volume"),
        functions.avg("Vol").alias("avg_volume"));

// 重命名列名
Dataset<Row> result = aggregatedData
    .withColumnRenamed("Yr", "year")
    .withColumnRenamed("M", "month")
    .withColumnRenamed("D", "day")
    .withColumnRenamed("HH", "hour");
```

### 3. 数据保存

```java
// 合并所有分区为一个分区
Dataset<Row> singlePartitionResult = result.coalesce(1);

// 保存结果到 CSV 文件
singlePartitionResult.write()
    .format("csv")
    .option("header", "true")
    .mode("overwrite")
    .save("processed_traffic_data");
```



## 节点和边的构建

### 1. 节点构建

- **节点构建**：`WktGeom` 包含了道路点的空间几何信息，作为每个节点的唯一标识。将 `WktGeom` 转化为字符串，然后使用哈希函数映射为节点索引 `[0, n-1]`。
- **节点信息获取**：`WktGeom` 包含了道路点的空间几何信息，利用这些信息可以确定图的节点。

### 2. 边构建

- **街道信息收集**：每个数据点包含三个街道信息：`street`（流量统计的街道）、`fromSt`（起始街道）和 `toSt`（终点街道）。

- **维护街道字典**：建立一个字典，键为街道名称，值为与该街道相关的所有节点的集合。

  ```python
  street_dict = {}
  for node in nodes:
      streets = [node['street'], node['fromSt'], node['toSt']]
      for street in streets:
          if street not in street_dict:
              street_dict[street] = set()
          street_dict[street].add(node['id'])
  ```

- **边的连接方式**：

  - 对于每一条街道，获取其相关的所有节点集，假设有 `m` 个节点。

  - **排序节点**：假设街道是直线的，根据节点的经度（longitude）对节点进行排序。

    ```python
    nodes_on_street = list(street_dict[street])
    nodes_on_street.sort(key=lambda node_id: node_longitude[node_id])
    ```

  - **连接节点**：按照排序后的顺序，将相邻的节点两两连接，即第一个节点连接第二个，第二个连接第三个，直到第 `m` 个节点。这样每条街道得到 `m-1` 条边，避免了生成过多的边。

    ```python
    for i in range(len(nodes_on_street) - 1):
        node_i = nodes_on_street[i]
        node_j = nodes_on_street[i + 1]
        edges.add((node_i, node_j))
    ```

### 3. 边权重计算

- **距离计算**：使用 `WktGeom` 中的坐标，直接计算节点之间的欧氏距离，作为边的权重。

  ```python
  # 示例代码
  from math import sqrt
  def euclidean_distance(node_a, node_b):
      x1, y1 = node_coordinates[node_a]
      x2, y2 = node_coordinates[node_b]
      return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
  ```

### 4. 生成图结构

- **图表示**：通过上述步骤，构建了一个包含节点和边的交通路网图，节点数为 `N`，邻接矩阵大小为 `N x N`。

- **数据整合**：将节点的时间信息进行合并，得到形状为 `(T, N, 1)` 的流量数据，其中 `T` 为时间长度，`N` 为节点数量。



## 模型构建与训练

- **核心模型**：采用先进的时空图神经网络模型对交通数据进行预测，捕获复杂的时空关联。
- **模型特点**：

  - **信号分离**：将交通信号分为扩散信号和固有信号，分别进行建模，提高预测精度。

  - **动态图学习**：引入动态图结构，捕捉交通网络的动态变化特性。
- **节点筛选策略**：

  - 由于完整图的节点数量较大，计算资源有限，选取流量最大的节点进行建模。

  - 具体步骤：

    - 选取流量最大的 10 个节点。

    - 从这 10 个节点出发，使用广度优先搜索（BFS），直到获得 500 个节点的子图。

    - 重新构建该子图的邻接矩阵，尺寸为 `507 x 507`。
- **预测任务**：

  - **短期预测**：预测未来 12 小时内的交通流量。

  - **长期预测**：预测未来 12 个月内的交通流量。

  - **差异**：主要在于时间维度的处理，短期关注细粒度变化，长期关注趋势变化。



## 前端展示模块

- **交通热力地图**：利用前端展示模块，将处理后的数据生成热力地图，直观展示纽约市不同区域的交通流量分布。

- **预测结果图**：提供实际流量和预测流量的对比图，展示模型的预测性能。

- **交互功能**：支持选择不同的时间段和区域，实时查看交通状况。

### 1.应用入口

 - app.py

  ```python
  @app.route("/", methods=["GET", "POST"])
  def index():
      # 处理请求，生成热力图
      # ...

      # 保存地图到文件
      nyc_map.save("./static/map.html")

      return render_template("index.html", ...)
  ```

### 2.模板文件

  - templates/index.html：主界面模板，包括时间选择、数据切换和下载按钮。

```html
<!-- 时间选择表单 -->
<form method="POST">
    <!-- 时间选择控件 -->
    <!-- ... -->
    <button type="submit">生成热力图</button>
</form>

<!-- 数据切换按钮 -->
<button onclick="switchToNow()">切换至实际示例</button>
<button onclick="switchToPredict()">切换至预测示例</button>

<!-- 热力图显示 -->
<iframe src="/static/map.html"></iframe>
```

  - templates/map.html：地图显示模板，使用 Folium 生成的交互式地图。

### 3.功能说明

  - **热力图生成**：根据用户选择的时间范围，生成对应的交通流量热力图，数据来源于处理后的 CSV 文件。

  - **数据切换**：通过 AJAX 请求切换实际数据和预测数据，刷新页面以显示不同的结果。

```javascript
    function switchToNow() {
        fetch("/switch_data_file", { ... }).then(() => window.location.reload());
    }
    
    function switchToPredict() {
        fetch("/switch_data_file", { ... }).then(() => window.location.reload());
    }
```

  - **热点街道数据下载**：提供前 50 个最热点街道的流量数据下载。

```python
    @app.route("/download_hot_streets", methods=["GET"])
    def download_hot_streets():
        # 生成热点街道数据
        # ...
        return send_file("static/hot_streets.csv", as_attachment=True)
```


https://github.com/user-attachments/assets/ce7cbb5d-c44f-4218-9793-93813ef36810

## 团队成员
| 姓名   | 学号        | 分工                                                    | 贡献百分比 |
| ------ | ----------- | ------------------------------------------------------- | ---------- |
| 郑智玮 | 51275903122 | 实验设计、仓库搭建、代码整合                            | 25%        |
| 金子龙 | 51275903071 | 实验设计、实现Sparks数据处理和web展示两个模块、代码优化 | 25%        |
| 麻旭晨 | 51275903113 | 实验设计、实现预测模块、代码优化                        | 25%        |
| 裘王辉 | 51275903106 | 实验设计、编写README                                    | 25%        |
