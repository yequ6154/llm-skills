import AVFoundation
import SwiftUI
import UIKit

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    var detectedFaces: [DetectedFace] = []

    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.previewLayer.session = session
        uiView.updateFaceOverlays(faces: detectedFaces)
    }
}

final class CameraPreviewUIView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    private let faceOverlayLayer = CAShapeLayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        faceOverlayLayer.strokeColor = UIColor.systemGreen.cgColor
        faceOverlayLayer.fillColor = UIColor.clear.cgColor
        faceOverlayLayer.lineWidth = 2
        layer.addSublayer(faceOverlayLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        faceOverlayLayer.frame = bounds
    }

    func updateFaceOverlays(faces: [DetectedFace]) {
        let path = UIBezierPath()

        for face in faces {
            // Vision 归一化坐标 → preview layer 坐标
            let converted = previewLayer.layerRectConverted(fromMetadataOutputRect: face.boundingBox)
            path.append(UIBezierPath(rect: converted))
        }

        faceOverlayLayer.path = path.cgPath
        faceOverlayLayer.strokeColor = (faces.contains(where: \.isPrimary)
            ? UIColor.systemGreen
            : UIColor.systemYellow).cgColor
    }
}
