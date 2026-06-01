import Foundation

/// 女友模式参数，与 specs/light-analysis-spec.yaml 对齐
enum GirlfriendModeConfig {
    // MARK: - 曝光 & 色调

    static let exposureBiasEV: Float = 0.3
    static let analysisIntervalFrames: Int = 3

    // MARK: - 光线分类阈值

    static let faceLumaMinGood: Double = 0.25
    static let faceToBgRatioMinGood: Double = 0.6
    static let faceToBgRatioMaxGood: Double = 1.4
    static let verticalGradientMaxGood: Double = 0.30
    static let leftRightDiffMaxGood: Double = 0.35

    static let faceToBgRatioBacklitMax: Double = 0.5
    static let verticalGradientTopLightMin: Double = 0.35
    static let leftRightDiffSideLightMin: Double = 0.40
    static let faceLumaLowLightMax: Double = 0.15

    // MARK: - UI 评分色阶

    static let lightScoreGreenMin: Int = 75
    static let lightScoreYellowMin: Int = 50

    // MARK: - 曝光滑条

    static let exposureSliderMinEV: Float = -1.0
    static let exposureSliderMaxEV: Float = 1.0

    /// 女友模式总 EV = 预设 bias + 用户手动补偿
    static func totalExposureBias(userAdjustment: Float) -> Float {
        exposureBiasEV + userAdjustment
    }
}
