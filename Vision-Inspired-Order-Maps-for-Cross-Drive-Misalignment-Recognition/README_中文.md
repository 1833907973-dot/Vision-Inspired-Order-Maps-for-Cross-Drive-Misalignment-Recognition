# Vision-Inspired Order Maps for Cross-Drive Misalignment Recognition

代码与部分数据

本包面向《Vision-Inspired Order Maps for Cross-Drive Misalignment Recognition》
所涉及的紧凑模型和跨电机迁移评估，材料来自已有 EPCDA 失效分析项目。

包含：核心 PyTorch 模型、数据读取与归一化、真实信号样本、原始结果表、
模型演示、结果均值和标准差重算脚本，以及原有模型测试。

## 使用方法

在解压目录运行：

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_results.py
python scripts/run_demo.py
python -m unittest discover -s tests
```

第一条分析命令核对九种方法、三个指标端点共 27 组结果，以及 24 项配对均值差；
对应的关键结果为 19.87% 和 54.34%。置信区间和 p 值保留已有统计输出，
不从少量示例数据重新计算。两项准确率来自不同监督条件，不能作直接性能比较。

模型演示使用随机初始化权重，用于检查数据到特征及分类输出的计算通路，
不把演示输出当作训练后的诊断结果。输出保存在 `results/`。

## 数据范围

144 个真实信号窗口，每个窗口 3 通道、128 个采样点，来自 144 个不同采集 run。
通道顺序为 rpm、iq、id，覆盖 200 W、400 W、1500 W 与五类状态。
样本从已有 8,540 个窗口中按功率和类别抽取，保留原有数值；不是模拟数据。
结果 CSV 来自完整实验，不能将示例子集视作产生这些结果的完整训练集。

## 版本关系

此处是部署通路 77,077 参数、323/333 Hz 数据的实现；TCE 稿件描述的
2.91 M 参数、10 kHz 实现不在本包中。材料来源见 `docs/PROVENANCE.md`。
本包只整理代码与部分数据，不修改论文，也不表示稿件已发表或已通过投稿审查。

在 GitHub 网页上传时，先解压，再上传目录内文件即可。公开仓库的首页将显示
`README.md`；本包没有代选代码或数据许可证。
