# DES Design Workbench 可下载版

本程序在用户自己的电脑上运行，页面地址为 `http://127.0.0.1:4173`，不依赖 ChatGPT 或 GPT 网站。

## Windows

1. 解压安装包到英文路径，例如 `D:\DES_Design_Workbench`。
2. 安装 Python 3.11 或 3.12，并勾选“Add Python to PATH”。
3. 双击 `INSTALL_AND_START_WINDOWS.bat`。首次运行会安装约数百 MB 的公开 Python 依赖，以后直接启动。
4. 浏览器打开后，点击 `Load validated example` 即可运行百里香酚–辛酸示例。

## Ubuntu / Linux

在解压目录执行 `bash install_and_start_linux.sh`，然后访问 `http://127.0.0.1:4173`。

## 可用范围

- 安装包自带百里香酚–辛酸的实验纯物性、两份 σ-profile，以及冻结的 B3/B4 混合物模型。
- 用户可输入其他体系的实验熔点和熔化焓，并选择 `γ = 1`、上传 `γ(x)` 或上传两份 σ-profile。
- 便携版不会自动下载或重新分发 ChemBERTa、TabPFN 等第三方大模型。因此，不在示例小库中的纯物性缺失时，用户需要手动输入实验值；完整研究环境可以另行安装经许可的纯物性模型包。
- 页面输出用于实验前筛选，不单独证明 DES 形成、相稳定性、安全性或文献新颖性。
