import Foundation
import Vision

final class FaceDetectionService {
    private let request: VNDetectFaceRectanglesRequest

    init() {
        request = VNDetectFaceRectanglesRequest()
    }

    func detect(in pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) -> [DetectedFace] {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])

        do {
            try handler.perform([request])
        } catch {
            return []
        }

        guard let results = request.results, !results.isEmpty else {
            return []
        }

        let primaryIndex = results.enumerated()
            .max(by: { $0.element.boundingBox.width < $1.element.boundingBox.width })?
            .offset

        return results.enumerated().map { index, observation in
            DetectedFace(
                boundingBox: observation.boundingBox,
                isPrimary: index == primaryIndex
            )
        }
    }
}
