# 人工智能开源软件开发与管理课程项目文档

## 4DNDF 论文总结

### 研究问题
在动态环境中构建准确的3D地图是自动驾驶和机器人技术的关键挑战。传统方法难以同时处理动态物体移除和静态地图重建，特别是在处理连续时间序列的LiDAR数据时。

### 创新点
1. **4D隐式神经表示**：提出了一种时空隐式神经地图表示方法，将4D场景编码为时间相关的截断符号距离函数(TSDF)
2. **动态静态分离**：通过时间相关的基函数权重，自然地分离静态背景和动态物体
3. **分段训练策略**：设计了高效的损失函数和数据采样策略，确保静态点监督的一致性

### 方法流程图

点云输入 → 隐式编码 → 多层感知机（MLP） → 距离场 SDF/NDF → 损失函数 → 反向传播 → 输出静态场/动态场

## 论文公式和程序代码对照表

| 论文公式 | 代码文件                                  | 行数范围            | 描述                                                         |
| -------- | ----------------------------------------- | ------------------- | ------------------------------------------------------------ |
| 公式(1)  | `model/DCTdecoder.py`                     | ~55-100             | 基函数权重与时间索引相乘，重建时空 TSDF（`Decoder.forward` 中 `signals = torch.mm(...)`) |
| 公式(2)  | `model/DCTdecoder.py`                     | ~25-54              | DCT 时间基函数初始化，构建 `full_basis` 以分离静态/动态分量  |
| 公式(7)  | `model/sdfloss.py`<br>`static_mapping.py` | ~8-45<br>~165-190   | 截断 SDF 面点监督（`sdfLoss`、`smooth_sdfLoss`）及训练循环中的近表面损失项 |
| 公式(8)  | `model/sdfloss.py`<br>`static_mapping.py` | ~70-150<br>~190-200 | 数值 Eikonal 梯度约束（`numerical_ekional_*`、`double_numerical_normals`）与训练阶段的 Eikonal 正则 |
| 公式(10) | `static_mapping.py`                       | ~185-195            | 自由空间截断约束 `tloss = |φ - τ|`，确保射线空段接近截断距离 |
| 公式(11) | `static_mapping.py`                       | ~195-205            | 确定自由空间损失，利用确定空域样本拉回静态场数值             |
| 公式(12) | `static_mapping.py`                       | ~205-215            | 总损失函数：surface、Eikonal、free-space、certain-free 的加权求和 |

## 安装说明

基于 NVIDIA RTX 5080Ti的Ubuntu 22.04系统

```bash
# 1. 克隆代码仓库
git clone https://github.com/PRBonn/4dNDF.git
cd 4dNDF

# 2. 创建并激活Python虚拟环境（推荐）
conda create -n 4dndf python=3.8
conda activate 4dndf

# 3. 安装PyTorch（请根据你的CUDA版本选择合适的安装命令）
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# 4. 装 PyTorch3D
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html

# 5. 装其他依赖
pip install open3d==0.17 scikit-image tqdm pykdtree plyfile
conda install -c conda-forge quaternion
```

## 运行说明

```
sh ./script/download_test_data.bash
```

- 这是一个 shell 脚本，用于 “Sanity test and Demo” 阶段。即下载一个小规模测试数据集（20 帧来自 KITTI 序列 00）用于快速跑通流程。
- 目的：先用一个很小的数据量确认环境、代码、依赖是否正确、流程是否正常。
- 运行后会在仓库下某个 `data/` 或 `output/` 目录看到这 20 帧数据。

```
python static_mapping.py config/test/test.yaml
```

- 使用 `static_mapping.py` 主入口脚本，并以 `config/test/test.yaml` 配置文件运行。
- 功能：以刚下载的测试数据为输入，执行“静态地图提取 + 动态物体分割”流程。运行完后，会生成静态 mesh（静态地图）并且可视化动态物体分割结果。 提到生成的 mesh 存在 `output/test`。 
- 可视化：按 “space” 键播放序列，黄色点表示静态部分，红色表示被识别为动态的点。
- 需要保证配置文件（`config/test/test.yaml`）里数据路径、模型超参、输出路径都正确。

```
sh ./script/download_cofusion.bash
```

- 下载用于 “Surface reconstruction” 评估用的数据集： Co‑Fusion 的 car4 数据集（已经转换成点云格式）
- 作用是获取一个更大、更真实的数据集用于重建静态地图的质量评估。

```
sh ./script/run_cofusion.bash
```

- 在下载好 Co-Fusion 数据集之后，运行该脚本启动 pipeline（即 `static_mapping.py`）＋评估脚本。
- 最终输出会存放在 `output/cofusion` 文件夹中。你可以查看生成的 mesh 并和 ground-truth 比对。

```
sh ./script/download_newer_college.bash
```

- 下载另一个用于重建评估的数据集： Newer College 数据（作者选取其中约 1300 帧）。

```
sh ./script/run_newer_college.bash
```

- 在下载之后运行 pipeline + 评估流程，用于 Newer College 数据集。输出放在 `output/newer_college`。 

```
sh ./script/download_baseline.bash
```

- 下载基线方法（baseline）已经重建的 mesh，以便你把作者的方法输出结果与这些基线结果做对比。
- 之后你可能需要修改 `eval/eval_cofusion.py` 或 `eval/eval_newercollege.py` 中的 `est_ply` 路径来指向你下载的基线 mesh。

Dynamic objects segmentation 部分：

- 下载用于动态物体分割评估的数据集： KTH_DynamicMap_Benchmark 的 KTH_dynamic。README 给出目录结构和说明。

- 编译 benchmark 的 repo：

  ```
  git clone --recurse-submodules https://github.com/KTH-RPL/DynamicMap_Benchmark.git
  cd DynamicMap_Benchmark/script
  mkdir build && cd build
  cmake ..
  make
  ```

- 复制 `4dNDF/eval/evaluate_single_kth.py` 到 benchmark 的脚本路径。

- 以序列 00 为例：

  ```
  python static_mapping.py config/kth/00.yaml
  # 模型训练／运行完后，生成静态点云: data/kth/00/static_points.pcd
  cd /path/to/DynamicMap_Benchmark/scripts/build/
  ./export_eval_pcd /path/to/4dNDF/data/KTH_dynamic/00 static_points.pcd 0.05
  python /your/path/to/DynamicMap_Benchmark/scripts/py/eval/evaluate_single_kth.py /path/to/4dNDF/data/KTH_dynamic/00
  ```

## 测试运行结果

运行后有以下几个结果

![image-20251124201203624](https://github.com/pecksn/4dNDF/blob/main/image/image-20251124201203624.png)

### 地表重建

#### 测试数据（来自KITTI seq 00的20帧）

![image-20251124204929401](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251124204929401.png)



![image-20251125140128827](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251125140128827.png)

#### Newer College数据集

##### **mesh.ply（完整网格）**

- 包含**整个扫描区域**的所有重建几何
- 可能包括边缘噪声、不完整的区域
- 文件相对较大
- 适合观察整体重建质量

![image-20251124204804326](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251124204804326.png)

##### **cropped_mesh.ply（裁剪网格）**

- 只保留**感兴趣区域(ROI)**

- 去除了边缘噪声和不相关区域

- 文件更小，加载更快

- 便于**集中分析关键区域**的重建质量

  ![image-20251124204736004](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251124204736004.png)

##### 评估测试结果

![image-20251125142928713](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251125142928713.png)

#### Co-Fusion的car4数据

![image-20251124204533781](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251124204533781.png)

##### 评估测试结果

![](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251125135639530.png)



### 动态对象分割

#### 序列00评估结果

![image-20251124204523175](D:\Desktop\GDTU\研一课程\人工智能开源软件\4dNDF\image\image-20251124204523175.png)
