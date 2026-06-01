import SwiftUI

struct CameraScreen: View {
    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if camera.permissionGranted {
                cameraContent
            } else {
                permissionPlaceholder
            }
        }
        .onAppear {
            if camera.permissionGranted {
                camera.configureAndStart()
            }
        }
        .onDisappear {
            camera.stop()
        }
        .onChange(of: camera.userExposureAdjustment) { _, _ in
            camera.applyExposureAdjustment()
        }
    }

    // MARK: - Camera Content

    private var cameraContent: some View {
        ZStack {
            CameraPreviewView(
                session: camera.session,
                detectedFaces: camera.detectedFaces
            )
            .ignoresSafeArea()

            VStack {
                topBar
                Spacer()
                bottomPanel
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
    }

    private var topBar: some View {
        VStack(spacing: 10) {
            HStack {
                Text("女友模式")
                    .font(.headline)
                    .foregroundStyle(.white)
                Spacer()
                phaseBadge
            }

            LightRadarView(result: camera.lightResult)
        }
    }

    private var phaseBadge: some View {
        Text("Phase 0")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.white.opacity(0.2), in: Capsule())
            .foregroundStyle(.white)
    }

    private var bottomPanel: some View {
        VStack(spacing: 12) {
            HintBannerView(hint: camera.lightResult.hint)

            ExposureSliderView(adjustment: $camera.userExposureAdjustment) {
                camera.applyExposureAdjustment()
            }

            debugMetrics
        }
    }

    /// Phase 0 调试信息，Alpha 阶段可隐藏
    private var debugMetrics: some View {
        Group {
            if camera.lightResult.detectedFaceCount > 0 {
                Text(String(
                    format: "脸 %.0f%% · 背景 %.0f%% · 比值 %.2f",
                    camera.lightResult.faceLuma * 100,
                    camera.lightResult.backgroundLuma * 100,
                    camera.lightResult.faceToBackgroundRatio
                ))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.white.opacity(0.5))
            }
        }
    }

    // MARK: - Permission Placeholder

    private var permissionPlaceholder: some View {
        VStack(spacing: 20) {
            Image(systemName: "camera.fill")
                .font(.system(size: 48))
                .foregroundStyle(.white.opacity(0.6))

            Text("需要相机权限")
                .font(.title2.bold())
                .foregroundStyle(.white)

            Text(camera.errorMessage ?? "女友模式需要访问相机来检测人脸和光线")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button("开启相机") {
                camera.checkPermission()
            }
            .buttonStyle(.borderedProminent)
            .tint(.pink)
        }
    }
}

#Preview {
    CameraScreen()
}
