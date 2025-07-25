# gemma-practice


* gemma是google发布的轻量级开源LLM
* LiteRT-LM是一个C++库，旨在高效地在各种设备（从手机到嵌入式系统）上运行语言模型流水线。
* Bazel是google开发的构建工具(build system)
* QAIRT: QAIRT是连接“AI模型”和“高通硬件（NPU）”之间的桥梁。
* QAIRT SDK: Qualcomm AI Runtime SDK
* Android NDK: native development kit for android, 是一个工具集，允许你使用 C 或 C++ 语言 为 Android 开发应用的一部分（比如性能要求高的模块）。是Android SDK(java/kotlin写的)的扩展。
<!-- * 高通8635 本质上是 SM8750 的降频/定制版，也就是 Snapdragon 8 Gen 2 的一个变种。架构和指令集完全兼容。核心布局、NPU、GPU 等模块几乎一致，只是主频略低。 -->

---

* `.so`文件: shared object file, 是Linux下的动态链接库文件，类似于Windows下的`.dll`文件。供多个程序共享使用的函数库，包含程序运行时需要的函数、类、和变量等。
* `LD_LIBRARY_PATH`: 是一个环境变量，告诉系统运行时去哪些路径查找`.so`文件。
  * 系统默认只在一些固定的目录下查找动态链接库文件（如 `/usr/lib`、`/usr/local/lib` 等），如果你的`.so`文件不在这些目录下，就需要设置 `LD_LIBRARY_PATH` 环境变量来指定额外的查找路径。

