import Foundation
import AppKit

class BoundingBoxView {
    let shapeLayer: CAShapeLayer
    let textLayer: CATextLayer
    
    init() {
        shapeLayer = CAShapeLayer()
        shapeLayer.fillColor = NSColor.clear.cgColor
        shapeLayer.lineWidth = 2
        shapeLayer.isHidden = true
        
        textLayer = CATextLayer()
        textLayer.foregroundColor = NSColor.black.cgColor
        textLayer.isHidden = true
        //textLayer.contentsScale = NSScreen.main.scale
        textLayer.fontSize = 14
        textLayer.font = NSFont(name: "SF Mono", size: textLayer.fontSize)
        //textLayer.alignmentMode = kCAAlignmentCenter
    }
    
    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(shapeLayer)
        parent.addSublayer(textLayer)
    }
    
    func show(frame: CGRect, label: String, color: NSColor) {
        CATransaction.setDisableActions(true)
        
        //let path = NSBezierPath(rect: frame)

        //shapeLayer.path = path.cgPath
        let path = CGMutablePath()
        path.addRect(frame)
        shapeLayer.path = path
        shapeLayer.strokeColor = color.cgColor
        shapeLayer.isHidden = false
        
        textLayer.string = label
        textLayer.backgroundColor = color.cgColor
        textLayer.isHidden = false
        
        let attributes = [
            NSAttributedString.Key.font: textLayer.font as Any
        ]
        
        let textRect = label.boundingRect(with: CGSize(width: 400, height: 100),
                                          options: .truncatesLastVisibleLine,
                                          attributes: attributes, context: nil)
        let textSize = CGSize(width: textRect.width + 12, height: textRect.height)
        let textOrigin = CGPoint(x: frame.origin.x-1, y: frame.origin.y + frame.height - textSize.height)
        textLayer.frame = CGRect(origin: textOrigin, size: textSize)
    }
    
    func hide() {
        shapeLayer.isHidden = true
        textLayer.isHidden = true
    }
}
