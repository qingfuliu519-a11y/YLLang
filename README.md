# YLLang (Your Local Language)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.x-green)](https://developer.nvidia.com/cuda-toolkit)

**YLLang** 是一个可本地部署的大模型推理项目，使用 **C++** 编写。它旨在提供一个高效、轻量级的推理引擎，支持在本地环境中运行大型语言模型，无需依赖云端服务。

---

## ✨ 特性

- 🚀 基于 **C++17** 实现，性能优越
- ⚡ 支持 **CUDA** 加速，充分利用 GPU 资源
- 🔧 使用 **libtorch**（PyTorch C++ API）进行模型加载与推理
- 🧠 集成 **LLVM**，提供潜在的即时编译（JIT）优化能力
- 🔒 完全本地部署，保护数据隐私

---

## 📋 前置要求

在编译和运行 YLLang 之前，请确保已安装以下依赖：

| 依赖 | 说明 | 下载链接 |
|------|------|----------|
| **CUDA** | GPU 加速核心库 | [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) |
| **cuDNN** | 深度神经网络加速库 | [cuDNN Archive](https://developer.nvidia.com/cudnn-archive) |
| **libtorch** | PyTorch C++ 库（建议使用 **Pre-cxx11 ABI** 版本） | [PyTorch 官网](https://pytorch.org/get-started/locally/) |
| **LLVM** | 编译器基础设施（用于 JIT 优化） | [LLVM Releases](https://github.com/llvm/llvm-project/releases) |

> **注意**：请确保各依赖的版本相互兼容，尤其是 CUDA 与 cuDNN 的版本匹配，并将LLVM加入到您的环境变量中。

---

## 🔧 编译与安装

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/YLLang.git
cd YLLang
```

### 2. 配置编译参数
通过 CMake 指定 libtorch 安装目录完成编译配置，将命令中的 `your TORCH lib dir` 替换为本地实际的 libtorch 路径（如 `/usr/local/libtorch` 或 `D:/libtorch`）：
```bash
cmake -DTORCH_INSTALL_DIR="your TORCH lib dir" .
```

### 3. 编译项目
执行编译命令完成项目构建，可通过 `-j` 参数指定线程数加速编译（推荐）：
```bash
# 基础编译
make
# 加速编译（使用8线程，可根据CPU核心数调整）
make -j8
# Linux/macOS 自动适配核心数
make -j$(nproc)
```

### 4. 验证安装
编译完成后，可运行测试程序验证环境是否配置成功：
```bash
# 运行基础测试
./yllang_test
```

> **常见问题说明**：
> - 若提示 "CUDA not found"：检查 `CUDA_HOME` 环境变量是否配置，或通过 `cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .` 指定CUDA编译器路径
> - 若提示 "libtorch not found"：确认 `TORCH_INSTALL_DIR` 路径正确，且libtorch版本与CUDA匹配
> - Windows系统：替换 `make` 为 `ninja` 或使用Visual Studio打开生成的解决方案编译

---

## 🚀 快速开始
编译完成后，可通过以下命令运行基础推理示例：
```bash
# 运行大模型推理示例（替换为实际模型路径）
./yllang_infer --model_path "path/to/your/model" --prompt "Hello, YLLang!"
```

---

## 📄 许可证
本项目采用 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献
欢迎提交 Issue 和 Pull Request 来帮助改进 YLLang，贡献前建议先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)（可选）。

---

## ❗ 免责声明
YLLang 仅用于学习和研究目的，请勿用于商业场景中未经授权的大模型推理，使用前请确保遵守相关模型的开源协议和法律法规。