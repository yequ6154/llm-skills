import Foundation
import SwiftUI

/// 光线类型，与 specs/light-analysis-spec.yaml 对齐
enum LightType: String, CaseIterable, Identifiable {
    case good
    case backlit
    case topLight
    case sideLight
    case lowLight
    case unknown

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .good: return "顺光"
        case .backlit: return "逆光"
        case .topLight: return "顶光"
        case .sideLight: return "侧光"
        case .lowLight: return "弱光"
        case .unknown: return "分析中"
        }
    }

    var hint: String {
        switch self {
        case .good: return "光线不错，可以拍了 ✓"
        case .backlit: return "逆光了，让她转身面向亮处"
        case .topLight: return "顶光显脸肿，移到有阴影的地方"
        case .sideLight: return "侧光不均匀，让她转半圈"
        case .lowLight: return "太暗了，靠近灯光或开闪光灯"
        case .unknown: return "对准人脸，正在分析光线…"
        }
    }

    var defaultScore: Int {
        switch self {
        case .good: return 85
        case .backlit: return 40
        case .topLight: return 45
        case .sideLight: return 50
        case .lowLight: return 30
        case .unknown: return 0
        }
    }
}

struct LightAnalysisResult: Equatable {
    let lightType: LightType
    let score: Int
    let faceLuma: Double
    let backgroundLuma: Double
    let faceToBackgroundRatio: Double
    let hint: String
    let detectedFaceCount: Int

    static let empty = LightAnalysisResult(
        lightType: .unknown,
        score: 0,
        faceLuma: 0,
        backgroundLuma: 0,
        faceToBackgroundRatio: 0,
        hint: LightType.unknown.hint,
        detectedFaceCount: 0
    )

    var scoreColor: Color {
        if score >= GirlfriendModeConfig.lightScoreGreenMin {
            return .green
        }
        if score >= GirlfriendModeConfig.lightScoreYellowMin {
            return .yellow
        }
        return .red
    }
}

struct DetectedFace: Equatable {
    let boundingBox: CGRect
    let isPrimary: Bool
}
