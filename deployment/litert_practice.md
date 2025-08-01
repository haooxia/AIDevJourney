# LiteRT practice

- [LiteRT practice](#litert-practice)
    - [PyTorch to LiteRT](#pytorch-to-litert)
      - [Details](#details)
    - [Generative API / LLM PyTorch -\> litert](#generative-api--llm-pytorch---litert)
      - [原理细节](#原理细节)
  - [Run LiteRT model on Android](#run-litert-model-on-android)
  - [LiteRT-LM](#litert-lm)
    - [LiteRT vs LiteRT-LM vs MediaPipe GenAI Tasks](#litert-vs-litert-lm-vs-mediapipe-genai-tasks)
  - [MediaPipe](#mediapipe)


1. AI Edge 提供了一些工具，可以将Tensorflow, PyTorch, JAX模型转换为
   LiteRT支持的FlatBuffers格式（`.tflite` file）。
2. Android常见部署模型
   1. 图像分类/目标检测: MobileNetV1/V2/V3, EfficientNet-lite, YOLOv5-nano, etc.
   2. LLM: qwen3, gemma-3, gemma3n, etc.

---

1. select a model
   1. use an existing LiteRT model: [link](https://www.kaggle.com/models?framework=tfLite)
   2. convert a model into a LiteRT model

![picture 1](../images/95afcfe65365c7313e14c7d301bed2117c2b0da57e0ff3c23edfc2aedd0569b0.png)  


![picture 2](../images/d4fb3fb2ff7feb0a7331653163ecbf6d0db81da4f0dad35ff458ae797f0d0a71.png)  


### PyTorch to LiteRT

AI Edge Model Conversion and Inference Demo

This example demonstrates how to:
- Load a pretrained ResNet18 model
- Preprocess an input image
- Run inference using the original PyTorch model
- Convert the model to AI Edge (TFLite) format using `ai_edge_torch`
- Run inference with the converted model
- Compare the results
- Export the model as a `.tflite` file

```python
# *step 0: Import required libraries
import ai_edge_torch              # AI Edge conversion toolkit
import torch             
import numpy                   
import torchvision               # Pretrained vision models and transforms
from PIL import Image

# *step 1: load a pretrained model
# Load a pretrained ResNet18 model from torchvision
# The 'weights' argument is the recommended modern approach (instead of pretrained=True)
model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
).eval()

# *step 2: Prepare Sample Input
sample_input = (torch.randn(1, 3, 224, 224),)

# *step 3: Load and Preprocess an Actual Image (optional)
img_path = "test.png"
input_image = Image.open(img_path).convert("RGB")
# Define image preprocessing pipeline
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])
# Apply transformation and add batch dimension
img_tensor = transform(input_image).unsqueeze(0)

# *step 4: Inference with the Original PyTorch Model
with torch.no_grad():
    output_pt = model(img_tensor)
    probs_pt = torch.nn.functional.softmax(output_pt, dim=1)
# Get predicted class and probability
pred_class_pt = torch.argmax(probs_pt).item()
print(f"Predicted class: {pred_class_pt} and the probability is {probs_pt[0][pred_class_pt].item()}")

# *step 5: Convert the Model to AI Edge (TFLite) Format
edge_model = ai_edge_torch.convert(model.eval(), sample_input)

# *step 6: Inference with the Converted Model
output_lite = edge_model(img_tensor)
from scipy.special import softmax
probs_lite = softmax(output_lite, axis=1)
pred_class_lite = numpy.argmax(probs_lite).item()
print(f"Predicted class from edge model: {pred_class_lite} and the probability is {probs_lite[0][pred_class_lite]}")

# *step 7: Compare Results
if numpy.allclose(
    output_pt.detach().numpy(),
    output_lite,
    atol=1e-5,  # Absolute tolerance
    rtol=1e-5   # Relative tolerance
):
    print("Inference result with PyTorch and TFLite was within tolerance")
else:
    print("Something wrong with PyTorch → TFLite conversion")

# *step 8: Export the Model as a .tflite File
edge_model.export('resnet.tflite') # about 45MB
```

> 部署后有关签名、量化的步骤参考：[link](https://github.com/google-ai-edge/ai-edge-torch/blob/main/docs/pytorch_converter/README.md#use-odml-torch-conversion-backend-experimental)

---

#### Details

第一层：

1. `ai_edge_torch.conver()`转换过程中需要sample inputs来进行**tracing模型转换**（以tuple形式传入）（和ONNX转换原理类似，参考`deployment.md`）
   1. tracing是一种模型转换方法：提供一个input，运行一次forward propagation，记录下本次传播用到的的计算图。
   2. if the PyTorch model receives 3 tensors as positional arguments, the `convert` function receives 1 tuple with 3 entries.
   3. convert expects a `torch.nn.Module` with a `forward function` that receives tensors as arguments and returns tensors as outputs

第二层：
`ai_edge_torch.convert()`API具体转换过程：

1. 模型导出：使用PyTorch2.x的`torch.export()`功能，追踪模型并生成一个中间表示（FX图）（ref: [link](https://developers.googleblog.com/en/ai-edge-torch-high-performance-inference-of-pytorch-models-on-mobile-devices/)
   1. `torch.export()`是TorchDynamo和TorchScript的演进版，可以将PyTorch模型导出为**中间表示**，导出模型用于部署和优化
2. 优化和转换：AI Edge Torch应用编译器优化，比如操作融合(op fusion)和constant folding，以提升性能。优化便可后将中间便是转换为tflite格式文件。即可用于LiteRT和MediaTask
3. > 对于llm或transformer-based model可能涉及额外的步骤(如量化，什么KV Cache优化等等)

---

此外，还有很多模型转换为LiteRT/`.tflite`的方法，比如(仅供了解)：

* ultralytics公司(YOLO的公司吧)的`ultralytics`包同样可以一行代码将YOLO model从torch转换为LiteRT格式: [link](https://medium.com/google-developer-experts/yolov10-to-litert-object-detection-on-android-with-google-ai-edge-2d0de5619e71)
  * PyTorch -> ONNX Graph -> TensorFlow SavedModel -> LiteRT
* Datature训练平台/公司可以自动导出为.tflite格式: [link](https://datature.io/blog/how-to-use-litert-for-real-time-inferencing-on-android)

### Generative API / LLM PyTorch -> litert

> reference:
> https://ai.google.dev/edge/litert/models/edge_generative?authuser=1
> https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative

* AI Edge Torch Generative API是一个针对移动设备优化的library，用于将transformer-based PyTorch models转为LiteRT格式，支持在设备上进行图像和文本生成。（eg Gemma, TinyLlama, qwen...）
* 目前仅支持CPU，并计划支持GPU和NPU。
* now is an *early developer preview*

TO BE CONTINUED...


---

[reference](https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative#model-authoring-using-edge-generative-api)

1. 你可以自选一个任意的训练好的PyTorch模型，可以来自hugginface, kaggle...
2. 使用[example](https://github.com/google-ai-edge/ai-edge-torch/blob/main/ai_edge_torch/generative/examples/README.md)中有了的模型，或者自己基于Edge Generative API去[re-author](https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative#model-authoring-using-edge-generative-api)一个模型...
3. 使用库中提供的Quantization API来量化模型，以减小size和提升性能
4. 使用模型评估管道验证模型实现和质量
5. 将PyTorch LLM转为`.tflite`格式
6. 设备端测部署

---

#### 原理细节

.pth/.safetensors -> .tflite

> step 1-3就是re-authoring重构的过程

1. 读取用户输入的参数(模型大小、路径、量化模式等参数)
2. 模型构建: 用AI Edge Torch的原生、优化过的层重新定义模型架构
   1. 定义架构: 描述model的所有细节，e.g., 层数、头数、激活函数等
   2. 实例化: 根据上述配置，构建一个空的、结构上已为edge设备优化的PyTorch实例(基于通用的`DecoderOnlyModel`)
3. 权重迁移
   1. 加载: 从下载的safetensors权重/.pth中加载原始权重
   2. 映射与转换: 依据 TensorNames 映射表，将原始权重（包括名称和必要的结构变换，如融合QKV权重）填充到第二步创建的空模型实例中。
   3. 得到一个与原始模型在数学上等价，但在结构上已深度优化的 PyTorch 模型。
4. 多签名(signature)导出：为不同的推理阶段(prefill/decode)定义并导出不同的计算图(FX Graph)
5. 图优化、量化与序列化：
   1. 图优化: 对导出的每个FX Graph应用一系列优化，如算子融合、死代码消除等。
   2. 量化: 在此阶段，TFLite转换器根据用户指定的量化配置 (QuantConfig)，在将计算图序列化为最终模型时，同步地对权重进行量化（如从FP32转为INT8），并插入必要的量化/反量化算子。
   3. 序列化: 将经过优化和量化的计算图最终写入到一个包含多个签名的高性能 .tflite 文件中

---

Reauthoring: 就是用AI Edge Torch提供的构建块重写一遍model。将为服务器设计的model，重新打造成为edge设备定制的高性能model。

**为什么reauthoring一遍就快了？**

* 原因：替换为简单算子、内存布局优化、消除动态逻辑、融合优化。（还是很抽象是吧，慢慢研究吧..）
* 其实就是把一个Pythonic(灵活但缓慢的model)，重建为一个”C++ like”(僵化但极致高效)的计算图，以榨干手机芯片的性能


## Run LiteRT model on Android

1. choose a prebuilt liteRT model from [link](https://ai.google.dev/edge/litert/models/trained?authuser=1)
2. convert other models to LiteRT format using AI Edge Torch

![picture 0](../images/356147e31efc5feb8518d608af330fd1b026397d6be2e7aafb3f729203a026bd.png)  

Android应用需要具备：
* LiteRT runtime environment for executing the model
  * LiteRT in **Google Play services runtime environment**
* Model input hander to transform data into tensors
* Model output handler to **receive** output result tensors and **interpret** them as predictions results

## LiteRT-LM

A C++ library to efficiently run language models across edge platforms.

> c++可以最大限度地榨取硬件性能，减少资源占用；可以精细控制系统底层、内存布局和执行流程；高度可移植，去目标平台重新编译即可运行。

LiteRT-LM提供的model是`.litertlm`格式

### LiteRT vs LiteRT-LM vs MediaPipe GenAI Tasks

**LiteRT vs. LiteRT-LM vs. MediaPipe GenAI Tasks**

* **LiteRT**是on-device runtime, 允许你将**individual** PyTorch, TF, JAX models转换为.tflite格式，并在on-device设备上运行
* **LiteRT-LM**为开发者提供pipline framework, 用于将multiple LiteRT models和pre and post processing componnents (e.g., tokenizer, vision encoder, text decoder)**拼接**在一块
* [MediaPipe GenAI Tasks](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)是开箱即用的native API，可以直接使用(支持kotlin/JS/Swift)，你只需要设置几个参数(e.g., temperature, topK)就可以运行语言模型
  * native API: 比如它可以提供原生kotlin的类和方法，你在android应用中直接调用该类和方法，传入文本，设置少量参数，直接拿结果即可
  * 你不理解/管理任何model和inference的流程，只需要传参拿结果


==为什么我android不直接用mediapipe GenAI Tasks呢？== 既然它[可行](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android)，问题出在哪儿了？

---


**`.tflite` vs. `.task` vs. `.litertlm`**

* `.tflite`是Google LiteRT使用的**通用**机器学习模型格式，包含模型执行图和可选的元数据。是FlatBuffer格式
  * 可以通过AI Edge Torch工具将PyTorch模型转换为`.tflite`格式，适合一般ML模型。当然还支持Tensorflow, JAX等框架
* `.task`: MediaPipe用该文件表示lm，该格式是一个zip，包含multiple LiteRT files, components, and metadata.
* `.litertlm`: 是`.task`格式的演进版本，包含了additional metadata且enable better compression
  * 不可直接转换，需要先用AI Edge Torch Generative API将PyTorch模型转换为.tflite，然后**可能通过**LiteRT-LM或MediaPipe的工具打包为.litertlm格式（继续研究，感觉也不是不行...
  * ==注意==：目前LiteRT-LM发布少量`.litertlm`文件，MediaPipe继续使用`.task`文件。等LiteRTLM发布first release，会让medapipe api使用`.litertlm`文件
  * > 这我得研究一下`.task`相关api啊

## MediaPipe

> LiteRT和MediaPipe Tasks都是Google的on-device AI框架，目标和场景有所不同。LiteRT更适合灵活性和自定义，后者更适合快速集成solution。

* MediaPipe Tasks是一个更高层次的即插即用的解决方案，开箱即用
  * 特点：提供端到端的解决方案，封装了比如从相机数据捕获、图像预处理、模型推理、后处理到结果呈现的整个流程。
    * 而且可能会封装多个模型吧
  * 核心是用cpp写的，所以同样的逻辑可以部署到Android/iOS/Web等多个平台
  * 允许一定程度的自定义（通过model maker）
* LiteRT是一个更底层、更纯粹的“AI模型推理引擎”
  * 不会包括上层逻辑，比如图像预处理后处理等解决方案，需要开发者自行实现
  * 所以如果你有自己的tflite模型，不需要再mediapipe的