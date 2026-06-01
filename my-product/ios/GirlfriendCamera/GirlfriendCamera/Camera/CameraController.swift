import AVFoundation
import Combine
import Foundation
import UIKit

final class CameraController: NSObject, ObservableObject {
    // MARK: - Published State

    @Published private(set) var isRunning = false
    @Published private(set) var permissionGranted = false
    @Published private(set) var lightResult: LightAnalysisResult = .empty
    @Published private(set) var detectedFaces: [DetectedFace] = []
    @Published var userExposureAdjustment: Float = 0 {
        didSet { exposureAdjustmentStorage = userExposureAdjustment }
    }
    @Published private(set) var errorMessage: String?

    /// 供 session 队列读取，避免 main.sync 死锁
    private var exposureAdjustmentStorage: Float = 0

    // MARK: - AVFoundation

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.myproduct.girlfriendcamera.session")
    private var videoDeviceInput: AVCaptureDeviceInput?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let analysisPipeline = FrameAnalysisPipeline()

    // MARK: - Lifecycle

    override init() {
        super.init()
        checkPermission()
    }

    func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { self.permissionGranted = true }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.permissionGranted = granted
                    if granted {
                        self?.configureAndStart()
                    } else {
                        self?.errorMessage = "需要相机权限才能使用女友模式"
                    }
                }
            }
        default:
            DispatchQueue.main.async {
                self.permissionGranted = false
                self.errorMessage = "请在「设置 → 隐私 → 相机」中开启权限"
            }
        }
    }

    func configureAndStart() {
        guard permissionGranted else { return }
        sessionQueue.async { [weak self] in
            self?.configureSession()
            self?.startSession()
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    func applyExposureAdjustment() {
        sessionQueue.async { [weak self] in
            self?.updateExposureBias()
        }
    }

    // MARK: - Session Configuration

    private func configureSession() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1280x720

        session.inputs.forEach { session.removeInput($0) }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            DispatchQueue.main.async { self.errorMessage = "无法访问后置摄像头" }
            return
        }
        session.addInput(input)
        videoDeviceInput = input

        session.outputs.forEach { session.removeOutput($0) }
        videoDataOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.setSampleBufferDelegate(self, queue: sessionQueue)

        guard session.canAddOutput(videoDataOutput) else {
            DispatchQueue.main.async { self.errorMessage = "无法配置视频分析输出" }
            return
        }
        session.addOutput(videoDataOutput)

        if let connection = videoDataOutput.connection(with: .video),
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }

        configureDevice(device)
    }

    private func configureDevice(_ device: AVCaptureDevice) {
        do {
            try device.lockForConfiguration()
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            applyExposureBias(to: device)
            device.unlockForConfiguration()
        } catch {
            DispatchQueue.main.async { self.errorMessage = "相机参数配置失败" }
        }
    }

    private func startSession() {
        guard !session.isRunning else { return }
        session.startRunning()
        DispatchQueue.main.async { self.isRunning = true }
    }

    // MARK: - Face-Priority Metering

    private func updateMetering(for face: DetectedFace) {
        guard let device = videoDeviceInput?.device else { return }

        let center = CGPoint(x: face.boundingBox.midX, y: face.boundingBox.midY)

        do {
            try device.lockForConfiguration()

            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = center
            }
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = center
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
            }

            applyExposureBias(to: device)
            device.unlockForConfiguration()
        } catch {
            // 静默失败，避免频繁 lock 冲突
        }
    }

    private func applyExposureBias(to device: AVCaptureDevice? = nil) {
        let targetDevice = device ?? videoDeviceInput?.device
        guard let targetDevice else { return }

        let userAdjustment = exposureAdjustmentStorage
        let bias = GirlfriendModeConfig.totalExposureBias(userAdjustment: userAdjustment)
        let clamped = min(
            max(bias, targetDevice.minExposureTargetBias),
            targetDevice.maxExposureTargetBias
        )

        do {
            try targetDevice.lockForConfiguration()
            targetDevice.setExposureTargetBias(clamped, completionHandler: nil)
            targetDevice.unlockForConfiguration()
        } catch {
            // ignore
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let orientation = Self.videoOrientation(for: connection)
        let (faces, lightResult) = analysisPipeline.process(
            pixelBuffer: pixelBuffer,
            orientation: orientation
        )

        if !faces.isEmpty, let primary = faces.first(where: \.isPrimary) {
            updateMetering(for: primary)
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            if !faces.isEmpty {
                self.detectedFaces = faces
            }
            if lightResult.lightType != .unknown {
                self.lightResult = lightResult
            }
        }
    }

    private static func videoOrientation(for connection: AVCaptureConnection) -> CGImagePropertyOrientation {
        if connection.videoRotationAngle == 90 {
            return .right
        }
        return .right
    }
}
