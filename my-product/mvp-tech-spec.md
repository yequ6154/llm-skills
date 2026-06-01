# MVP 功能清单 & 技术方案（iOS / Android）

> 版本：v0.1 MVP  
> 目标：8–10 周可演示的内测版，验证「女友满意度 ↑、重拍次数 ↓」

---

## 一、MVP 功能清单

### 1.1 功能优先级总览

| 优先级 | 功能 | 用户价值 | MVP 是否包含 |
|:------:|------|----------|:------------:|
| **P0** | 人脸检测 + 人脸优先测光 | 解决脸黑/脸过曝 | ✅ |
| **P0** | 光线类型识别 + 一句话提示 | 解决不懂对光 | ✅ |
| **P0** | 「女友模式」一键预设 | 零配置出片 | ✅ |
| **P0** | 拍前 3 秒质检 | 减少废片 | ✅ |
| **P0** | 基础相机（前后摄、闪光灯、变焦） | 可用性底线 | ✅ |
| **P1** | 构图辅助线（全身/半身模板） | 解决构图差 | ⏳ v0.2 |
| **P1** | 连拍智能选片 | 提高出片率 | ⏳ v0.2 |
| **P1** | 语音反馈（「再低一点」） | 解放双手 | ⏳ v0.2 |
| **P1** | 自然美颜（磨皮/肤色） | 女生满意度 | ⏳ v0.2 |
| **P2** | 场景预设包（餐厅/海边/夜景） | 差异化 | ❌ |
| **P2** | 拍后 AI 补光 | 救场兜底 | ❌ |
| **P2** | 远程指导（女友看取景框） | 社交功能 | ❌ |

### 1.2 P0 功能详细说明

#### F1：人脸优先测光（Face-Priority Metering）

**用户故事**：作为不会拍照的男生，我希望 App 自动以女友的脸为曝光基准，这样逆光时她不会脸黑。

**验收标准**：
- [ ] 预览画面实时以最大/中心人脸为测光区域
- [ ] 逆光场景：人脸亮度 ≥ 画面平均亮度 1.2x（可调）
- [ ] 支持多张人脸时，优先对焦最近/最大人脸
- [ ] 曝光补偿滑块：仅「亮一点 / 暗一点」两档或连续滑条

**不做（MVP）**：对焦/曝光分离的高级手势

---

#### F2：光线类型识别 + 提示（Light Radar）

**用户故事**：作为男生，我希望 App 告诉我「光好不好、怎么改」，而不是让我学摄影。

**识别类型**：

| 光线类型 | 检测逻辑（简化） | 提示文案示例 |
|----------|-----------------|-------------|
| 顺光 | 人脸亮度正常，背景不过曝 | 「光线不错，可以拍了 ✓」 |
| 逆光 | 背景亮、人脸暗，人脸/背景亮度比 < 0.5 | 「逆光了，让她转身面向亮处」 |
| 顶光 | 人脸上半部亮、眼窝/鼻下阴影重 | 「顶光显脸肿，移到有阴影的地方」 |
| 侧光 | 脸左右亮度差 > 40% | 「侧光不均匀，让她转半圈」 |
| 弱光 | 整体亮度低、噪点预估高 | 「太暗了，靠近灯光或开闪光灯」 |

**UI**：
- 取景框顶部：光线评分 0–100 + 颜色（红/黄/绿）
- 底部：一句话 actionable 提示

**验收标准**：
- [ ] 5 种光线类型识别准确率 ≥ 75%（内测人工标注）
- [ ] 提示延迟 < 500ms（每 3 帧分析一次，非每帧）

---

#### F3：「女友模式」预设（Girlfriend Mode）

**用户故事**：打开 App 就是最好用的模式，不需要选任何设置。

**预设参数（默认值，可 A/B 调整）**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 曝光补偿 | +0.3 EV（相对人脸测光） | 女生偏好略亮 |
| 色温 | 略暖 (+200K) | 肤色友好 |
| 对比度 | 略降 | 减少硬阴影 |
| 饱和度 | 略升 | 气色更好 |
| HDR | 自动开启 | 逆光保细节 |
| 人像虚化 | 2x 以上变焦时开启（若硬件支持） | 非必需 |

**验收标准**：
- [ ] 首次打开即为女友模式，无 onboarding 也可直接拍
- [ ] 预设可通过远程配置热更新（Firebase Remote Config 等）

---

#### F4：拍前质检（Pre-Shutter QA）

**用户故事**：按快门前 App 帮我检查会不会废片，避免被对象骂。

**检查项**：

| 检查项 | 阈值 | 反馈 |
|--------|------|------|
| 人脸清晰度 | 模糊度 score < 阈值 | 「有点糊，手稳一点」 |
| 人脸过曝 | 脸部高光 > 95% | 「脸太亮了，点『暗一点』」 |
| 人脸欠曝 | 脸部亮度 < 20% | 「脸太暗，靠近亮处」 |
| 人脸裁切 | 头顶/下巴出框 | 「退后一步，拍全身」 |
| 闭眼 | 双眼 EAR < 阈值 | 「眨眼了，再拍一张」（连拍时） |

**交互**：
- 软拦截：黄色警告，仍可强制拍摄
- 不硬阻断（避免用户烦躁）

**验收标准**：
- [ ] 快门按下前 200ms 内完成检查
- [ ] 废片率（内测标注）比系统相机降低 30%+

---

#### F5：基础相机能力

- [ ] 前后摄像头切换
- [ ] 1x / 2x 变焦（数码变焦可接受）
- [ ] 闪光灯：关 / 开 / 自动
- [ ] 照片保存至系统相册
- [ ] 快门音效（可关）
- [ ] 权限引导（相机、相册）

---

### 1.3 MVP 明确不做

- 社交分享、账号系统
- 订阅付费（内测免费）
- 复杂滤镜库
- 视频录制
- 云端 AI 补光
- Android 全机型适配（先主流旗舰）

---

### 1.4 成功指标（内测 KPI）

| 指标 | 目标 | 测量方式 |
|------|------|----------|
| 女友满意度 | ≥ 4.0 / 5.0 | 拍后即时问卷 |
| 重拍次数 | 比系统相机 ↓ 40% | 会话内计数 |
| 废片率 | ↓ 30% | 女生标记「这张不行」 |
| 男生 NPS | ≥ 30 | 内测结束问卷 |
| 单次拍摄耗时 | ≤ 系统相机 1.2x | 从打开到满意出片 |

---

## 二、技术架构总览

```
┌─────────────────────────────────────────────────────────┐
│                      UI Layer                           │
│  CameraPreview │ LightRadar │ HintBanner │ ShutterBtn   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   Camera Controller                       │
│  Session管理 │ 曝光控制 │ HDR开关 │ 变焦 │ 闪光灯        │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                  Vision Pipeline (端侧)                   │
│  FaceDetection │ LightAnalysis │ QualityCheck │ BlurDetect│
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                  Image Processing                         │
│  FaceMetering │ ToneMapping │ ColorGrade │ (PortraitBlur) │
└─────────────────────────────────────────────────────────┘
```

**设计原则**：
1. **端侧优先**：人脸检测、光线分析、质检全部 on-device，原图不上传
2. **Pipeline 异步**：Vision 分析与预览解耦，不阻塞 30fps 预览
3. **双端逻辑统一**：Kotlin Multiplatform (KMP) 或共享 C++ Core 封装算法

---

## 三、iOS 技术方案

### 3.1 技术栈

| 层级 | 选型 | 理由 |
|------|------|------|
| 语言 | Swift 5.9+ | 原生相机性能最优 |
| UI | SwiftUI + UIViewRepresentable | 相机预览需 UIKit 桥接 |
| 相机 | **AVFoundation** (`AVCaptureSession`) | 标准方案，可控曝光 |
| 人脸检测 | **Vision** (`VNDetectFaceRectanglesRequest`) | 系统级、低延迟 |
| 人脸 Landmark | Vision `VNDetectFaceLandmarksRequest` | 闭眼检测、肤质区域 |
| 图像分析 | Core Image + Metal | 亮度/色温/对比度实时分析 |
| HDR | `photoOutput.maxPhotoQualityPrioritization` + 多帧（Phase 2） | iOS 17+ 支持更好 |
| 人像虚化 | AVFoundation `builtInDualCamera` / Depth Data（有限） | MVP 可跳过，v0.2 加入 |
| 配置下发 | Firebase Remote Config | 预设参数热更新 |

### 3.2 核心实现要点

#### 相机 Session 配置

```swift
// 伪代码示意
let session = AVCaptureSession()
session.sessionPreset = .photo

// 输入
let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
try device?.lockForConfiguration()

// 人脸优先曝光：持续调整 exposurePointOfInterest
device?.exposureMode = .continuousAutoExposure
device?.exposurePointOfInterest = faceCenter  // 归一化坐标 0-1

// 女友模式默认补偿
device?.setExposureTargetBias(0.3, completionHandler: nil)

device?.unlockForConfiguration()
```

#### Vision Pipeline（每 3 帧执行）

```swift
let faceRequest = VNDetectFaceRectanglesRequest { request, error in
    guard let faces = request.results as? [VNFaceObservation] else { return }
    let primaryFace = faces.max(by: { $0.boundingBox.width < $1.boundingBox.width })
    // → 更新 exposurePointOfInterest
    // → 触发 LightAnalysis
}

let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
try handler.perform([faceRequest])
```

#### 光线分析（LightAnalysis）

```swift
// 基于 CIAreaAverage 分析人脸区域 vs 背景区域亮度
func analyzeLighting(face: CGRect, frame: CVPixelBuffer) -> LightType {
    let faceLuma = averageLuma(in: face, frame: frame)
    let bgLuma = averageLuma(outside: face, frame: frame)
    let ratio = faceLuma / max(bgLuma, 0.01)

    if ratio < 0.5 { return .backlit }
    if faceTopLuma - faceBottomLuma > threshold { return .topLight }
    if leftRightDiff > 0.4 { return .sideLight }
    if faceLuma < 0.15 { return .lowLight }
    return .good
}
```

#### 拍前质检

- 在 `AVCapturePhotoOutput.capturePhoto` 前，用最后一帧 preview buffer 做快速检查
- 闭眼：Vision Face Landmarks → 计算 Eye Aspect Ratio (EAR)
- 模糊：Laplacian variance on face crop

### 3.3 iOS 限制与应对

| 限制 | 影响 | 应对 |
|------|------|------|
| 第三方 App 无法调用 Photonic Engine 全部能力 | HDR 效果弱于系统相机 | 多帧合成放 v0.2；MVP 靠曝光补偿 + tone mapping |
| `Depth Effect` 仅系统相机完整 | 人像虚化弱 | MVP 不做虚化，或用 ML 分割（v0.2） |
| 后台 CPU 限制 | 分析不能太重 | 降采样到 720p 做 Vision |
| App Store 审核 | 相机权限说明清晰 | Info.plist `NSCameraUsageDescription` 写明用途 |

### 3.4 iOS 最低支持

- **iOS 16.0+**（Vision 性能足够）
- 推荐体验：**iPhone 13 及以上**

---

## 四、Android 技术方案

### 4.1 技术栈

| 层级 | 选型 | 理由 |
|------|------|------|
| 语言 | Kotlin | 官方推荐 |
| UI | Jetpack Compose | 现代 UI；CameraX 有 Compose 支持 |
| 相机 | **CameraX** (1.3+) | 简化生命周期；Google 维护 |
| 人脸检测 | **ML Kit Face Detection** 或 MediaPipe Face Mesh | 端侧、免费、低延迟 |
| 图像分析 | RenderScript 替代 → **RenderEffect / OpenGL ES** | 亮度分析 |
| HDR | CameraX `HDR` Extension（设备支持时） | 需检测 `ExtensionMode.HDR` |
| 美颜 | ML Kit + 自定义 shader（v0.2） | MVP 仅色调不对脸做几何变形 |
| 配置下发 | Firebase Remote Config | 与 iOS 统一 |

### 4.2 核心实现要点

#### CameraX 配置

```kotlin
// 伪代码示意
val preview = Preview.Builder()
    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
    .build()

val imageAnalysis = ImageAnalysis.Builder()
    .setTargetResolution(Size(1280, 720))
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build()

imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
    processFrame(imageProxy)  // 人脸检测 + 光线分析
    imageProxy.close()
}

val imageCapture = ImageCapture.Builder()
    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
    .build()

// 曝光补偿
camera.cameraControl.setExposureCompensationIndex(compensationIndex)
```

#### ML Kit 人脸检测

```kotlin
val options = FaceDetectorOptions.Builder()
    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)  // 闭眼检测
    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
    .build()

val detector = FaceDetection.getClient(options)
detector.process(inputImage)
    .addOnSuccessListener { faces ->
        val primary = faces.maxByOrNull { it.boundingBox.width() }
        updateMeteringRegion(primary)
        analyzeLighting(primary, imageProxy)
    }
```

#### CameraX Metering

```kotlin
val factory = SurfaceOrientedMeteringPointFactory(width, height)
val point = factory.createPoint(faceCenterX, faceCenterY)
val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AE)
    .setAutoCancelDuration(3, TimeUnit.SECONDS)
    .build()
camera.cameraControl.startFocusAndMetering(action)
```

### 4.3 Android 限制与应对

| 限制 | 影响 | 应对 |
|------|------|------|
| 厂商 ROM 差异大 | 曝光行为不一致 | 先适配 Pixel、Samsung、小米旗舰；维护设备兼容表 |
| CameraX HDR Extension 并非所有设备支持 | HDR 效果参差 | 运行时检测；不支持则软件 tone mapping |
| ML Kit 在不同芯片上性能差异 | 低端机卡顿 | MVP 最低骁龙 778G / 天玑 8100 |
| 权限碎片化 | Android 13+ 需细分权限 | 运行时请求 `CAMERA` |

### 4.4 Android 最低支持

- **Android 10 (API 29)+**
- 推荐体验：**Android 12+，8GB RAM 及以上**

---

## 五、跨端共享策略

### 5.1 方案对比

| 方案 | 优点 | 缺点 | 建议 |
|------|------|------|------|
| **各端独立实现** | 最快出 MVP | 算法漂移、维护双倍 | ✅ MVP 阶段推荐 |
| **KMP 共享逻辑层** | 光线分析、质检逻辑统一 | 相机层仍需原生 | v0.2 迁移 |
| **C++ Core + JNI/FFI** | 性能最优 | 开发成本高 | 规模化后再考虑 |

### 5.2 MVP 阶段：统一规格文档

两端实现必须对齐的 **共享 spec**（非代码共享）：

```yaml
# light-analysis-spec.yaml
backlit:
  face_to_bg_ratio: < 0.5
  hint: "逆光了，让她转身面向亮处"
top_light:
  vertical_gradient: > 0.35
  hint: "顶光显脸肿，移到有阴影的地方"
# ...

girlfriend_mode:
  exposure_bias: 0.3
  warmth_k: 200
  contrast: -0.1
  saturation: 0.05
```

---

## 六、项目结构建议

```
my-product/
├── readme.md
├── competitive-comparison.md
├── mvp-tech-spec.md
├── specs/
│   └── light-analysis-spec.yaml      # 跨端对齐 spec
├── ios/                               # Phase 2 代码
│   ├── GirlfriendCamera/
│   │   ├── App/
│   │   ├── Camera/                   # AVFoundation 封装
│   │   ├── Vision/                   # 人脸 + 光线 + 质检
│   │   └── UI/
│   └── GirlfriendCamera.xcodeproj
└── android/                           # Phase 2 代码
    ├── app/
    │   ├── camera/                   # CameraX 封装
    │   ├── vision/                   # ML Kit + 分析
    │   └── ui/
    └── build.gradle.kts
```

---

## 七、开发里程碑

| 阶段 | 周期 | 交付物 | 关键验证 |
|------|------|--------|----------|
| **Phase 0：技术验证** | 2 周 | 命令行/最小 Demo：相机预览 + 人脸测光 + 光线评分 | 逆光人脸亮度提升可见 |
| **Phase 1：MVP Alpha** | 3 周 | 可安装 App：女友模式 + 光线提示 + 拍前质检 | 5 对内测用户试用 |
| **Phase 2：MVP Beta** | 3 周 | iOS + Android 双端；相册保存；Remote Config | 20 对情侣 A/B 测试 |
| **Phase 3：迭代** | 2 周 | P1 功能（构图线、连拍选片）+ bug fix | KPI 达标 |

**总工期：约 10 周**

---

## 八、关键技术风险 & 预案

| 风险 | 概率 | 预案 |
|------|------|------|
| 第三方 App HDR 弱于系统相机 | 高 | 降低预期；强调「人脸曝光 + 引导」而非纯画质 |
| 光线分类准确率不足 | 中 | 简化为 3 类（好/一般/差）+ 具体建议 |
| Android 碎片化 | 高 | MVP 仅支持 5–10 款主流机型 |
| 美颜「假脸」反馈 | 中 | MVP 不做几何美颜，只做曝光/色调 |
| App 体积/发热 | 中 | Vision 分析降采样；限制分析频率 |

---

## 九、MVP 技术选型速查

| 能力 | iOS | Android |
|------|-----|---------|
| 相机框架 | AVFoundation | CameraX 1.3+ |
| 人脸检测 | Vision Framework | ML Kit Face Detection |
| 图像处理 | Core Image + Metal | OpenGL ES / RenderEffect |
| 曝光控制 | `exposurePointOfInterest` + `setExposureTargetBias` | `FocusMeteringAction` + `setExposureCompensationIndex` |
| HDR | `maxPhotoQualityPrioritization` | CameraX HDR Extension |
| 配置热更新 | Firebase Remote Config | Firebase Remote Config |
| 最低系统 | iOS 16 / iPhone 13+ | Android 10 / 骁龙 778G+ |

---

## 十、下一步行动

1. **Phase 0 启动**：iOS 先做 2 周技术验证（Apple 相机 API 成熟度更高）
2. **编写 `specs/light-analysis-spec.yaml`**：锁定跨端算法参数
3. **招募 5 对内测**：验证光线提示文案是否「男生看得懂、女生觉得有用」
4. **Parallel Android**：iOS Alpha 完成后启动 Android 移植
