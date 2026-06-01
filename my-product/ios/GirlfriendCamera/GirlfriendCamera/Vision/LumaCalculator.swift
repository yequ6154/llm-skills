import CoreVideo
import Foundation

/// 从 CVPixelBuffer 计算区域平均亮度（Rec.709 luma）
enum LumaCalculator {
    static func averageLuma(in normalizedRect: CGRect, pixelBuffer: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return 0
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)

        // Vision 坐标系：原点在左下，转为像素坐标（原点左上）
        let pixelRect = CGRect(
            x: normalizedRect.origin.x * CGFloat(width),
            y: (1 - normalizedRect.origin.y - normalizedRect.height) * CGFloat(height),
            width: normalizedRect.width * CGFloat(width),
            height: normalizedRect.height * CGFloat(height)
        ).integral

        guard pixelRect.width > 1, pixelRect.height > 1 else { return 0 }

        let startX = max(0, Int(pixelRect.minX))
        let startY = max(0, Int(pixelRect.minY))
        let endX = min(width, Int(pixelRect.maxX))
        let endY = min(height, Int(pixelRect.maxY))

        var sum: Double = 0
        var count: Int = 0

        // 降采样：每隔 4 像素采样，降低 CPU 开销
        let step = 4

        if pixelFormat == kCVPixelFormatType_32BGRA {
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            for y in stride(from: startY, to: endY, by: step) {
                let row = buffer.advanced(by: y * bytesPerRow)
                for x in stride(from: startX, to: endX, by: step) {
                    let offset = x * 4
                    let b = Double(row[offset])
                    let g = Double(row[offset + 1])
                    let r = Double(row[offset + 2])
                    sum += 0.2126 * r + 0.7152 * g + 0.0722 * b
                    count += 1
                }
            }
        } else if pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
                    pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
            guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 1,
                  let yPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else {
                return 0
            }
            let yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            let yBuffer = yPlane.assumingMemoryBound(to: UInt8.self)
            for y in stride(from: startY, to: endY, by: step) {
                let row = yBuffer.advanced(by: y * yBytesPerRow)
                for x in stride(from: startX, to: endX, by: step) {
                    sum += Double(row[x])
                    count += 1
                }
            }
            if count > 0 {
                return (sum / Double(count)) / 255.0
            }
            return 0
        }

        guard count > 0 else { return 0 }
        return (sum / Double(count)) / 255.0
    }

    /// 计算人脸区域垂直梯度：(上半 - 下半) / 均值
    static func verticalGradient(for faceRect: CGRect, pixelBuffer: CVPixelBuffer) -> Double {
        let topHalf = CGRect(
            x: faceRect.origin.x,
            y: faceRect.origin.y + faceRect.height * 0.5,
            width: faceRect.width,
            height: faceRect.height * 0.5
        )
        let bottomHalf = CGRect(
            x: faceRect.origin.x,
            y: faceRect.origin.y,
            width: faceRect.width,
            height: faceRect.height * 0.5
        )
        let topLuma = averageLuma(in: topHalf, pixelBuffer: pixelBuffer)
        let bottomLuma = averageLuma(in: bottomHalf, pixelBuffer: pixelBuffer)
        let mean = max((topLuma + bottomLuma) / 2, 0.01)
        return (topLuma - bottomLuma) / mean
    }

    /// 计算左右脸亮度差
    static func leftRightDifference(for faceRect: CGRect, pixelBuffer: CVPixelBuffer) -> Double {
        let leftHalf = CGRect(
            x: faceRect.origin.x,
            y: faceRect.origin.y,
            width: faceRect.width * 0.5,
            height: faceRect.height
        )
        let rightHalf = CGRect(
            x: faceRect.origin.x + faceRect.width * 0.5,
            y: faceRect.origin.y,
            width: faceRect.width * 0.5,
            height: faceRect.height
        )
        let leftLuma = averageLuma(in: leftHalf, pixelBuffer: pixelBuffer)
        let rightLuma = averageLuma(in: rightHalf, pixelBuffer: pixelBuffer)
        let mean = max((leftLuma + rightLuma) / 2, 0.01)
        return abs(leftLuma - rightLuma) / mean
    }

    /// 背景亮度：人脸框外扩 1.5 倍区域与人脸框之间的环形近似（简化：全帧减去人脸）
    static func backgroundLuma(excluding faceRect: CGRect, pixelBuffer: CVPixelBuffer) -> Double {
        let fullFrame = CGRect(x: 0, y: 0, width: 1, height: 1)
        let fullLuma = averageLuma(in: fullFrame, pixelBuffer: pixelBuffer)
        let faceLuma = averageLuma(in: faceRect, pixelBuffer: pixelBuffer)
        // 简化估算：背景 ≈ 全帧与人脸的加权差（Phase 0 够用）
        let faceArea = faceRect.width * faceRect.height
        let bgWeight = max(1 - faceArea, 0.1)
        return max((fullLuma - faceLuma * faceArea) / bgWeight, 0.01)
    }
}
