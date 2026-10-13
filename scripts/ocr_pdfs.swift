// macOS-only offline OCR for scanned corpus PDFs. Original PDFs are never changed.
// Run: swift scripts/ocr_pdfs.swift data/BCTC.pdf data/<scanned-file>.pdf
import Foundation
import PDFKit
import Vision
import CryptoKit

struct OCRPage: Codable {
    let page: Int
    let text: String
    let mean_confidence: Float
}
struct OCRDocument: Codable {
    let source_sha256: String
    let engine: String
    let recognition_languages: [String]
    let pages: [OCRPage]
}
enum OCRError: Error {
    case message(String)
}

func recognize(_ url: URL) throws {
    let output = URL(fileURLWithPath: url.path + ".ocr.json")
    if FileManager.default.fileExists(atPath: output.path) {
        throw OCRError.message("Sidecar already exists: \(output.lastPathComponent). Inspect/remove it explicitly before a rerun.")
    }
    let pdfBytes = try Data(contentsOf: url)
    guard let document = PDFDocument(data: pdfBytes), document.pageCount > 0 else {
        throw OCRError.message("Cannot open PDF: \(url.lastPathComponent)")
    }
    let languages = ["vi-VT", "en-US"]
    var pages = [OCRPage]()
    for number in 0..<document.pageCount {
        let result: OCRPage = try autoreleasepool {
            guard let page = document.page(at: number) else {
                throw OCRError.message("Missing PDF page \(number + 1)")
            }
            let bounds = page.bounds(for: .mediaBox)
            let scale: CGFloat = 3.0
            let width = Int(ceil(bounds.width * scale))
            let height = Int(ceil(bounds.height * scale))
            guard let context = CGContext(data: nil, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: 0,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                throw OCRError.message("Could not render PDF page")
            }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.scaleBy(x: scale, y: scale)
            context.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
            page.draw(with: .mediaBox, to: context)
            guard let image = context.makeImage() else {
                throw OCRError.message("Could not rasterize PDF page")
            }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.recognitionLanguages = languages
            request.usesLanguageCorrection = true
            try VNImageRequestHandler(cgImage: image).perform([request])
            let candidates = (request.results ?? []).compactMap { $0.topCandidates(1).first }
            let text = candidates.map { $0.string }.joined(separator: "\n")
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw OCRError.message("OCR produced no text on page \(number + 1)")
            }
            let confidence = candidates.reduce(Float(0)) { $0 + $1.confidence } / Float(candidates.count)
            return OCRPage(page: number + 1, text: text, mean_confidence: confidence)
        }
        pages.append(result)
        print("\(url.lastPathComponent): \(number + 1)/\(document.pageCount) pages OCR complete")
    }
    let hash = SHA256.hash(data: pdfBytes).map { String(format: "%02x", $0) }.joined()
    let documentResult = OCRDocument(source_sha256: hash, engine: "Apple Vision VNRecognizeTextRequest",
                                     recognition_languages: languages, pages: pages)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(documentResult).write(to: output, options: .atomic)
    print("Saved \(output.path)")
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.isEmpty {
    fputs("Usage: swift scripts/ocr_pdfs.swift <scanned.pdf> [...]\n", stderr)
    exit(2)
}
do {
    for argument in arguments { try recognize(URL(fileURLWithPath: argument)) }
} catch {
    fputs("OCR failed: \(error)\n", stderr)
    exit(1)
}
