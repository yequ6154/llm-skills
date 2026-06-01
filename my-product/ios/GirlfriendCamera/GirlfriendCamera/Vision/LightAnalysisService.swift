import CoreVideo
import Foundation
import Vision

/// 光线类型识别 + 评分，遵循 light-analysis-spec.yaml
final class LightAnalysisService {
    func analyze(pixelBuffer: CVPixelBuffer, faces: [DetectedFace]) -> LightAnalysisResult {
        guard let primaryFace = faces.first(where: \.isPrimary) ?? faces.first else {
            return .empty
        }

        let faceRect = primaryFace.boundingBox
        let faceLuma = LumaCalculator.averageLuma(in: faceRect, pixelBuffer: pixelBuffer)
        let bgLuma = LumaCalculator.backgroundLuma(excluding: faceRect, pixelBuffer: pixelBuffer)
        let ratio = faceLuma / max(bgLuma, 0.01)
        let verticalGradient = LumaCalculator.verticalGradient(for: faceRect, pixelBuffer: pixelBuffer)
        let leftRightDiff = LumaCalculator.leftRightDifference(for: faceRect, pixelBuffer: pixelBuffer)

        let lightType = classify(
            faceLuma: faceLuma,
            ratio: ratio,
            verticalGradient: verticalGradient,
            leftRightDiff: leftRightDiff
        )

        let score = computeScore(
            lightType: lightType,
            faceLuma: faceLuma,
            ratio: ratio,
            verticalGradient: verticalGradient,
            leftRightDiff: leftRightDiff
        )

        return LightAnalysisResult(
            lightType: lightType,
            score: score,
            faceLuma: faceLuma,
            backgroundLuma: bgLuma,
            faceToBackgroundRatio: ratio,
            hint: lightType.hint,
            detectedFaceCount: faces.count
        )
    }

    private func classify(
        faceLuma: Double,
        ratio: Double,
        verticalGradient: Double,
        leftRightDiff: Double
    ) -> LightType {
        if faceLuma <= GirlfriendModeConfig.faceLumaLowLightMax {
            return .lowLight
        }
        if ratio <= GirlfriendModeConfig.faceToBgRatioBacklitMax {
            return .backlit
        }
        if verticalGradient >= GirlfriendModeConfig.verticalGradientTopLightMin {
            return .topLight
        }
        if leftRightDiff >= GirlfriendModeConfig.leftRightDiffSideLightMin {
            return .sideLight
        }
        if faceLuma >= GirlfriendModeConfig.faceLumaMinGood,
           ratio >= GirlfriendModeConfig.faceToBgRatioMinGood,
           ratio <= GirlfriendModeConfig.faceToBgRatioMaxGood,
           verticalGradient <= GirlfriendModeConfig.verticalGradientMaxGood,
           leftRightDiff <= GirlfriendModeConfig.leftRightDiffMaxGood {
            return .good
        }
        // 未完全满足 good 条件但无明显问题时，给 intermediate good
        if ratio > GirlfriendModeConfig.faceToBgRatioBacklitMax && faceLuma > 0.2 {
            return .good
        }
        return .sideLight
    }

    private func computeScore(
        lightType: LightType,
        faceLuma: Double,
        ratio: Double,
        verticalGradient: Double,
        leftRightDiff: Double
    ) -> Int {
        var score = Double(lightType.defaultScore)

        switch lightType {
        case .good:
            score = 75
            score += min(faceLuma * 30, 15)
            score += min((1 - abs(ratio - 1)) * 20, 10)
        case .backlit:
            score = max(50 - (GirlfriendModeConfig.faceToBgRatioBacklitMax - ratio) * 40, 10)
        case .topLight:
            score = max(55 - verticalGradient * 30, 10)
        case .sideLight:
            score = max(60 - leftRightDiff * 40, 15)
        case .lowLight:
            score = max(faceLuma / GirlfriendModeConfig.faceLumaLowLightMax * 40, 5)
        case .unknown:
            score = 0
        }

        return min(max(Int(score), 0), 100)
    }
}
