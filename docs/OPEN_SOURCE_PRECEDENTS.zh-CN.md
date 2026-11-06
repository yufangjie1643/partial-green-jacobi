# 相近拓扑优化论文的源码发布形式

核对日期：2026-11-06。先在作者 Zotero 文库定位论文和附件，再检查论文中的可用性声明、作者仓库及可访问的归档记录。本表是领域相近论文的实践比较，不是对目标期刊 *Mechanics Based Design of Structures and Machines* 的强制政策解释。

| 文献／期刊 | 已核实的发布形式 | 可以借鉴的做法 |
|---|---|---|
| Zhang 等，2023，*Structural and Multidisciplinary Optimization*，[论文](https://doi.org/10.1007/s00158-023-03657-y) | [homo3d](https://github.com/lavenklau/homo3d) 提供源码、依赖环境、编译说明、不同目标和体积分数的调用示例；当前 README 明确主分支正在重构，论文复现使用 `paper-version` 分支 | 提供短小运行入口；将论文版本与后续开发版本分开 |
| Träff 等，2023，*CMAME*，[论文](https://doi.org/10.1016/j.cma.2023.116043) | Futhark、OpenMP-GPU、OpenMP-CPU 和非线性实现分别归档；[Futhark 的 Zenodo v0.9.1](https://doi.org/10.5281/zenodo.7791871) 明确对应论文版本，并链接 GitHub 同名标签 | 不仅给仓库地址，还标明确切版本；不同实现单独说明环境和用途 |
| Li 等，2026，*Advances in Engineering Software*，[论文](https://doi.org/10.1016/j.advengsoft.2026.104244) | [SCTOP](https://github.com/lichaoBIT/SCTOP) 采用紧凑 MATLAB 程序，README 给出函数接口、参数含义和一行示例，仓库包含 MIT 许可证 | 一个读者可直接复制运行的最小例子非常重要；参数命名应对应论文 |
| Ladecký 等，2026，*CMAME*，[论文](https://doi.org/10.1016/j.cma.2026.119269) | 参考文献 [45] 引用 Zenodo 补充材料 DOI `10.5281/zenodo.20446919`；正文数据声明为按需提供。本次没有成功读取该 Zenodo 记录 | 补充材料归档与完整原始数据公开是两回事；可用性声明必须具体列明范围 |

Träff 等其他实现的归档为：[OpenMP-GPU](https://doi.org/10.5281/zenodo.7791868)、[OpenMP-CPU](https://doi.org/10.5281/zenodo.7791870)、[非线性 Futhark](https://doi.org/10.5281/zenodo.7791869)。论文的数据声明称未使用数据，但实际代码有独立归档；因此仅查看声明段落不足以判断复现材料是否齐全。

## 本研究采用的组织形式

1. **GitHub 源码仓库**：保留核心求解器、CUDA 外层、比较方法、参数冻结文件、CPU 表格复算和最小 GPU 示例。
2. **论文对应版本**：使用 `v0.1.0` 标签、Release、提交号和源文件哈希标明当前稿件的实现。后续算法改动使用新版本。
3. **数据分包**：开发扫描、独立回放、完整流程记录分别提供压缩附件；避免把 GB 级密度和位移场写入 Git 历史。
4. **结果到代码的对应表**：区分扫描／标定、冻结状态分析、完整拓扑优化和外部求解器消融，明确各自入口和计时口径。
5. **许可证与来源**：项目代码采用 MIT，保留 SCTOP 的相关声明；muGrid 按 LGPL-3.0 单独获取，不将其标作本项目 MIT 代码。

GitHub Release 是当前实际发布渠道。Zenodo 可以在后续归档这个已经固定的版本并分配 DOI；本次没有生成或填写尚不存在的 DOI。

Zotero 来源条目：Zhang `J4YJ53U8`，Träff `7E7QD3RE`，Li `SLEUPDXM`，Ladecký `9ZPHVBP8`。期刊 PDF 和个人文库内容不随仓库再分发。
