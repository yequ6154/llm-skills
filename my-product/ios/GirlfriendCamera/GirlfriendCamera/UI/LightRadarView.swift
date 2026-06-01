import SwiftUI

struct LightRadarView: View {
    let result: LightAnalysisResult

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .stroke(result.scoreColor.opacity(0.3), lineWidth: 4)
                    .frame(width: 52, height: 52)

                Circle()
                    .trim(from: 0, to: CGFloat(result.score) / 100)
                    .stroke(result.scoreColor, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .frame(width: 52, height: 52)
                    .rotationEffect(.degrees(-90))

                Text("\(result.score)")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(result.scoreColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("光线雷达")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(result.lightType.displayName)
                    .font(.headline)
                if result.detectedFaceCount > 0 {
                    Text("检测到 \(result.detectedFaceCount) 张人脸")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct HintBannerView: View {
    let hint: String

    var body: some View {
        Text(hint)
            .font(.subheadline.weight(.medium))
            .multilineTextAlignment(.center)
            .foregroundStyle(.white)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14))
    }
}

struct ExposureSliderView: View {
    @Binding var adjustment: Float
    var onChange: () -> Void

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Text("暗一点")
                    .font(.caption)
                Spacer()
                Text(String(format: "%+.1f EV", adjustment + GirlfriendModeConfig.exposureBiasEV))
                    .font(.caption.monospacedDigit())
                Spacer()
                Text("亮一点")
                    .font(.caption)
            }
            .foregroundStyle(.white.opacity(0.8))

            Slider(
                value: $adjustment,
                in: GirlfriendModeConfig.exposureSliderMinEV...GirlfriendModeConfig.exposureSliderMaxEV,
                step: 0.1
            ) { editing in
                if !editing { onChange() }
            }
            .tint(.yellow)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

#Preview {
    VStack(spacing: 16) {
        LightRadarView(result: LightAnalysisResult(
            lightType: .backlit,
            score: 42,
            faceLuma: 0.18,
            backgroundLuma: 0.65,
            faceToBackgroundRatio: 0.28,
            hint: LightType.backlit.hint,
            detectedFaceCount: 1
        ))
        HintBannerView(hint: LightType.backlit.hint)
    }
    .padding()
    .background(Color.gray)
}
