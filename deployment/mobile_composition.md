# Mobile Composition

- [Mobile Composition](#mobile-composition)
  - [Mobile Hardware Composition](#mobile-hardware-composition)
    - [SoC (System on Chip)](#soc-system-on-chip)
      - [SoC Manufacturer](#soc-manufacturer)
      - [CPU](#cpu)
      - [GPU](#gpu)
      - [NPU](#npu)
        - [NPU在手机中的应用](#npu在手机中的应用)
        - [CDSP (QC)](#cdsp-qc)
      - [DSP](#dsp)
      - [PCB (Printed Circuit Board)](#pcb-printed-circuit-board)
    - [RAM](#ram)
    - [ROM](#rom)
    - [Screen](#screen)
    - [Camera](#camera)
    - [Battery](#battery)
    - [Other](#other)
  - [Mobile Software Composition](#mobile-software-composition)
    - [Operating System](#operating-system)
    - [App Ecosystem](#app-ecosystem)

> rough introduction, to be continue...

## Mobile Hardware Composition

### SoC (System on Chip)

SoC芯片：含有多个功能块，如CPU, GPU, NPU, 共享内存Memory Cache, 数字信号处理器DSP（处理扬声器、麦克风、传感器等）, 图像信号处理器ISP（处理摄像头拍摄的图像和视频等）, 内存控制器等等。

#### SoC Manufacturer

**知名SoC制造厂商：**

* **联发科MediaTek/MTK**: **天玑Dimensity系列**和**Helio系列**
  * 性价比高吧，中低端市场强势
  * 24年市场占比~35%
* **高通Qualcomm**: **骁龙Snapdragon**系列
  * 性能强、兼容好，广泛应用于Android手机
  * 代表型号：高端骁龙8系列（如Snapdragon 8 Gen 3）、中端骁龙7系列、入门骁龙6/4系列
  * 24年市场占比~27%
* **苹果Apple**: **A系列**和**M系列**
  * 自研，性能领先
  * A系列用于iPhone，M系列用于Mac
  * 24年市场占比~16%
* **紫光展锐Unisoc**: **虎贲Tiger**系列
  * 主要用于入门、中低端Android手机
  * 24年市场占比~14%
* **三星Samsung**: **Exynos系列**
  * 自研，主要用于三星手机
  * 24年市场占比~4%
* **华为海思HiSilicon**: **麒麟Kirin系列**
  * 自研，主要用于华为手机
  * 24年市场占比~4%

---

**SoC used by Moto:**

| 产品线 | 主要 SoC 厂商 | 主要芯片系列 |
| :--- | :--- | :--- |
| **旗舰系列 (Edge, Razr)** | 高通 (Qualcomm)，少数机型用联发科 | 骁龙 8 系列、天玑 9000/8000 系列 |
| **中端系列 (Moto G 高端型号)** | 高通 (Qualcomm) 和 联发科 (MediaTek) | 骁龙 7/6 系列、天玑 7000/1000/800 系列 |
| **入门/预算系列 (Moto G/E 低端型号)** | 联发科 (MediaTek) 和 紫光展锐 (UNISOC) | 联发科 Helio G/P 系列、展锐虎贲 T 系列 |

> searched by Gemini, may be not accurate, but the general idea is correct. (need to verify it later)

#### CPU

CPU(Central Processing Unit)主处理器，手机的**大脑**/核心。负责执行操作系统、应用程序的指令，处理通用计算任务、逻辑控制、数据调度等。
* **通用性强**，什么都能干，**但不见得每件事都最快**。
* 举例： 高通骁龙Kryo核心、苹果A系列芯片中的CPU核心、联发科天玑Cortex核心。

#### GPU

GPU (Graphics Processing Unit) 图形处理器，手机的“**视觉引擎**”，主要负责图像和视频的渲染、处理。它擅长**并行处理大量的简单重复运算**。

* 高度并行化，非常适合图形渲染、视频编解码、游戏
* 浮点运算能力强
* 举例： 高通骁龙Adreno GPU、苹果A系列芯片中的GPU核心、ARM Mali GPU。

#### NPU

NPU (Neural Processing Unit) 神经网络处理器，手机的“**AI大脑**”，专门用于加速AI和ML任务。主要负责执行**神经网络模型的推理运算**。

==**为何SoC有了CPU和GPU还需要NPU？**==

* 神经网络计算的核心是大量的矩阵乘法和卷积运算，这些运算是高度并行的，但又**不完全等同于GPU擅长的图形渲染**。
* **功耗**：如果所有的AI任务都交给CPU或GPU来处理，功耗会非常大。**专门设计的NPU执行AI任务时功耗远低于二者**，符合手机对续航的极高要求。

> 其实电脑也正在集成NPU/独立AI加速卡(e.g., google的TPU)

NPU frequence: NPU的时钟频率，以MHz或GHz为单位。
* NPU频率越高，意味着NPU每秒能处理的数据越多，AI匀速那速度越快。e.g., NPU频率为800MHz，说明NPU每秒钟可进行8亿次的基本运算周期

##### NPU在手机中的应用

* 计算摄影
  * AI场景识别： 自动识别拍摄场景（如美食、人像、夜景），并优化参数。
  * 人像模式/虚化： 精准识别人物边缘并进行背景虚化。
  * 超级夜景：多帧合成，降噪，提升暗光照片质量。
  * AI美颜： 自然地优化面部特征。
  * image super-resolution, image inpainting, etc.
* 语音识别与处理：
  * 智能语音助手： 离线语音识别，提高响应速度和准确性。
  * 实时翻译： 加速语音和文本翻译。
* 人脸识别与解锁：
  * 3D人脸识别： 实现更安全、快速的解锁。
  * 表情识别： 应用于AR表情、虚拟形象等。
* AR/VR应用：
  * 手势识别： 识别用户手势进行交互。

##### CDSP (QC)

高通CDSP： Compute Digital Signal Processor，在高通芯片中有时也称为HVX或Hexagon DSP。

对于SM8750而言，你在SoC中看不到一个显式叫"NPU"的模块，但Hexagon DSP + Tensor Accelerator本质上就是高通的NPU实现。即CDSP承担了NPU的功能。

所以对于这类机器(没有独立的NPU)而言，我们可以认为CDSP就是NPU。

* Burst爆发模式：CDSP短时间内以其最高或接近最高性能运行，能迅速完成特定任务。
* Balence平衡模式：CDSP在功耗、散热和持续性能之间找到一个最佳平衡点

频率（时钟速度）是衡量性能的一个重要指标：处理器每秒执行的时钟周期数，单位MHz或GHz。
理论上，频率越高，处理器在单位时间内执行的指令越多，因此性能越强。

所以在测试llm的prefill speed/decode speed时候，应该在burst和balance两种模式下进行测试，得到更为全面的性能画像。

#### DSP

DSP (Digital Signal Processor) 数字信号处理器
和通用CPU相比，它在执行数学运算、实时处理等方面更高效，特别适用于音频、视频、图像、雷达、通信、控制系统等领域。

#### PCB (Printed Circuit Board)

PCB硬刷电路板是一切电子元器件的载体，它将**SoC芯片**、内存芯片、无线通信模块、电池等等元器件连接起来，形成一个完整的电路系统。

> 理解为电脑的主板。


### RAM 

RAM (Random Access Memory) 随机存取存储器，**暂存**当前正在运行的数据和程序。

DDR: Double Date Rate Synchronous Dynamic RAM （双倍数据率同步动态随机存取存储器）

DDR frequence: 内存的时钟频率，以MHz或GHz为单位。
* DDR频率越高，意味着内存传输速度越快，数据读取和写入效率更高

### ROM

ROM (Read-Only Memory) 只读存储器，**永久存储**手机的**操作系统**、应用程序和用户数据。

### Screen

屏幕是主要的输出设备。
分类：
- LCD (Liquid Crystal Display) 液晶显示屏
- OLED (Organic Light Emitting Diode) 有机发光二极管显示屏
- AMOLED (Active Matrix Organic Light Emitting Diode) 主动矩阵有机发光二极管显示屏
分辨率（如FHD+、QHD+）、刷新率（如60Hz、120Hz）是关键指标。

### Camera

摄像头是手机的主要输入设备之一。

### Battery

电池是手机的主要电源供应设备。

Power Efficiency（功耗）: 手机的功耗效率，通常以mAh（毫安时）或Wh（瓦时）来衡量。

### Other

* 通信模块：用于实现手机与外部网络的连接，包括4G/5G、Wi-Fi、蓝牙等。
* 传感器Sensors：包括**加速度计**、**陀螺仪**、**光线传感器**、距离传感器、指纹传感器、人脸识别模块等。
* 音频模块：包括扬声器、麦克风、耳机接口/Type-C音频输出等。
* 其他结构：外壳、散热系统、按钮、充电接口等。

## Mobile Software Composition

### Operating System

操作系统是手机的核心软件，负责管理硬件资源和提供用户界面。

* Android：基于**Linux内核**的**开源**操作系统，广泛应用于各种手机品牌。由**Google主导**开发和维护。
  * 厂商定制系统：各手机厂商（如小米、华为、OPPO等）会在Android系统上进行定制，添加自己的UI界面和功能。如ColorOS（OPPO）、MIUI（小米）、One UI（三星）。
  * AOSP（Android Open Source Project）：Android的开源项目，提供了Android系统的基础代码和框架。
* iOS：**Apple**公司开发的**闭源**操作系统，专门用于iPhone、iPad等设备。
* HarmonyOS：华为开发的操作系统，旨在替代Android，支持多种设备（手机、平板、智能家居等）。
  * > 正在逐步去安卓化，大多数华为手机用的“鸿蒙”系统仍然兼容安卓，是“安卓+鸿蒙混合形态”；但未来的“鸿蒙NEXT”将完全摆脱安卓，成为真正独立的国产系统。

市场份额
全球：Android ~**74**%, iOS ~22%, HarmonyOS ~4%
中国：Android ~64%, iOS ~17%, HarmonyOS ~**19**%
美国：Android ~**35**%, iOS ~**65**%

### App Ecosystem

应用生态系统是手机软件的核心组成部分，包括各种预装应用程序和服务。如应用商店（如Google Play、App Store）、社交媒体应用（如微信、QQ、微博）、视频播放应用（如YouTube、Netflix）等。