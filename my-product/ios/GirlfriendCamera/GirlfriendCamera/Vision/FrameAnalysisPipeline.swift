import AVFoundation
import Foundation

/// 协调人脸检测 + 光线分析，按 spec 间隔帧执行
final class FrameAnalysisPipeline {
    private let faceDetection = FaceDetectionService()
    private let lightAnalysis = LightAnalysisService()
    private var frameCounter = 0

    func process(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> (faces: [DetectedFace], lightResult: LightAnalysisResult) {
        frameCounter += 1
        guard frameCounter % GirlfriendModeConfig.analysisIntervalFrames == 0 else {
            return ([], .empty)
        }

        let faces = faceDetection.detect(in: pixelBuffer, orientation: orientation)
        let lightResult = lightAnalysis.analyze(pixelBuffer: pixelBuffer, faces: faces)
        return (faces, lightResult)
    }

    func reset() {
        frameCounter = 0
    }
}
