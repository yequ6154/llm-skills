# GirlfriendCamera — iOS Phase 0

Phase 0 技术验证：**相机预览 + 人脸优先测光 + 光线雷达评分**。

## 快速开始

### 环境要求

- macOS + Xcode 15+
- iOS 16.0+ 真机（**相机功能必须在真机测试**，模拟器无摄像头）
- Apple Developer 账号（免费账号即可真机调试）

### 打开项目

```bash
open my-product/ios/GirlfriendCamera/GirlfriendCamera.xcodeproj
```

### 运行步骤

1. 用 USB 连接 iPhone
2. Xcode → Target **GirlfriendCamera** → **Signing & Capabilities**
3. 选择你的 **Team**（Personal Team 即可）
4. 顶部设备选你的 iPhone
5. `Cmd + R` 运行
6. 首次启动允许相机权限

## 项目结构

```
GirlfriendCamera/
├── App/
│   └── GirlfriendCameraApp.swift      # 入口
├── Camera/
│   ├── CameraController.swift         # AVFoundation 会话 + 人脸测光
│   └── CameraPreviewView.swift        # 预览层 + 人脸框 overlay
├── Vision/
│   ├── FaceDetectionService.swift     # Vision 人脸检测
│   ├── LightAnalysisService.swift     # 光线类型 + 评分
│   ├── LumaCalculator.swift           # 区域亮度计算
│   └── FrameAnalysisPipeline.swift    # 每 3 帧分析一次
├── Models/
│   ├── LightType.swift                # 光线类型 & 结果模型
│   └── GirlfriendModeConfig.swift     # 与 spec yaml 对齐的参数
├── UI/
│   ├── CameraScreen.swift             # 主界面
│   └── LightRadarView.swift           # 光线雷达 + 提示 + 曝光滑条
├── Assets.xcassets
└── Info.plist
```

## Phase 0 已实现

| 功能 | 状态 | 说明 |
|------|------|------|
| 相机预览 | ✅ | AVFoundation 720p |
| 人脸检测 | ✅ | Vision Framework |
| 人脸优先测光 | ✅ | exposurePointOfInterest + 女友模式 +0.3 EV |
| 光线类型识别 | ✅ | 顺光/逆光/顶光/侧光/弱光 |
| 光线评分 0–100 | ✅ | 红/黄/绿三色 |
| 一句话提示 | ✅ | 底部 Hint Banner |
| 曝光滑条 | ✅ | 暗一点 ↔ 亮一点 |
| 人脸框 overlay | ✅ | 绿色=主脸 |
| 拍照/相册 | ❌ | Phase 1 |
| 拍前质检 | ❌ | Phase 1 |

## 验证清单

在以下场景测试，观察光线评分与提示是否合理：

- [ ] **顺光**：人脸面向窗户，评分应 ≥ 75（绿色）
- [ ] **逆光**：人物背对亮背景，应提示「逆光了…」，人脸自动提亮
- [ ] **顶光**：正午户外，应提示顶光
- [ ] **弱光**：暗处，应提示靠近灯光
- [ ] **曝光滑条**：拖动后预览亮度变化

## 算法参数

阈值与文案定义在 `../../specs/light-analysis-spec.yaml`，Swift 侧镜像于 `GirlfriendModeConfig.swift`。修改 spec 后需同步 Swift 常量。

## 已知限制（Phase 0）

1. **无拍照能力** — 仅预览与分析，Phase 1 加入 `AVCapturePhotoOutput`
2. **HDR 弱于系统相机** — 第三方 App 无法调用 Apple 全部计算摄影管线
3. **背景亮度为估算值** — 简化算法，Phase 1 可改为环形采样
4. **仅后置摄像头** — Phase 1 加入前后切换
5. **debug 指标可见** — Alpha 阶段可隐藏底部 luma 数值

## 下一步（Phase 1）

- [ ] `AVCapturePhotoOutput` 拍照 + 保存相册
- [ ] 拍前质检（模糊/过曝/欠曝/裁切）
- [ ] 前后摄像头切换
- [ ] 移除 debug 指标，加快门按钮 UI
