//
//  ViewController.swift
//  GestureRecognizer
//
//  Created by zero on 2020/12/13.
//

import UIKit
import CoreMotion
import CoreML

class ViewController: UIViewController {

    // MARK: Configuration
    static let samplesPerSecond = 25.0
    static let numberOfFeatures = 6
    static let windowSize = 20
    static let windowOffset = 5
    
    static let numberOfWindows = windowSize / windowOffset
    static let bufferSize = windowSize + windowOffset * (numberOfWindows - 1)
    static let sampleSizeAsBytes = ViewController.numberOfFeatures * MemoryLayout<Double>.stride
    static let windowOffsetAsBytes = ViewController.windowOffset * sampleSizeAsBytes
    static let windowSizeAsBytes = ViewController.windowSize * sampleSizeAsBytes
    
    static private func makeMLMultiArray(numberOfSamples: Int) -> MLMultiArray? {
        try? MLMultiArray(
            shape: [1, numberOfSamples, numberOfFeatures] as [NSNumber],
            dataType: .double
        )
    }
    
    
    
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    
    var classifier: GestureClassifier!
    var classifierOutput: GestureClassifierOutput!
    
    let motionManager = CMMotionManager()
    let queue = OperationQueue()
    let modelInput: MLMultiArray! = makeMLMultiArray(numberOfSamples: windowSize)
    let dataBuffer: MLMultiArray! = makeMLMultiArray(numberOfSamples: bufferSize)
    var bufferIndex = 0
    var isDataAvailable = false
    
    override func viewDidLoad() {
        super.viewDidLoad()

        do {
            classifier = try GestureClassifier(configuration: MLModelConfiguration())
        } catch {
            fatalError("Failed to create request")
        }
        
        // enable motion updates
        motionManager.deviceMotionUpdateInterval = 1 / ViewController.samplesPerSecond
        motionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: queue,
            withHandler: self.motionUpdatesHandler
        )
    }
    
    func addToBuffer(_ idx1: Int, _ idx2: Int, _ data: Double) {
        let index = [0, idx1, idx2] as [NSNumber]
        dataBuffer[index] = NSNumber(value: data)
    }

    func buffer(_ motionData: CMDeviceMotion) {
        for offset in [0, ViewController.windowSize] {
            let index = bufferIndex + offset
            if index >= ViewController.bufferSize {
                continue
            }
            addToBuffer(index, 0, motionData.rotationRate.x)
            addToBuffer(index, 1, motionData.rotationRate.y)
            addToBuffer(index, 2, motionData.rotationRate.z)
            addToBuffer(index, 3, motionData.userAcceleration.x)
            addToBuffer(index, 4, motionData.userAcceleration.y)
            addToBuffer(index, 5, motionData.userAcceleration.z)
        }
        bufferIndex = (bufferIndex + 1) % ViewController.windowSize
    }
    
    func motionUpdatesHandler(data motionData: CMDeviceMotion?, error: Error?) {
        guard let motionData = motionData else {
            let errorText = error?.localizedDescription ?? "Unknown"
            print("Device motion update error: \(errorText)")
            return
        }

        buffer(motionData)
        if bufferIndex == 0 {
            isDataAvailable = true
        }
        process()
    }
 
    func process() {
        if isDataAvailable
            && bufferIndex % ViewController.windowOffset == 0
            && bufferIndex + ViewController.windowOffset <= ViewController.windowSize {
    
            let window = bufferIndex / ViewController.windowOffset
            memcpy(
                modelInput.dataPointer,
                dataBuffer.dataPointer.advanced(by: window * ViewController.windowOffsetAsBytes),
                ViewController.windowSizeAsBytes
            )

            var classifierInput: GestureClassifierInput! = nil
            if classifierOutput != nil {
                classifierInput = GestureClassifierInput(
                    features: modelInput,
                    hiddenIn: classifierOutput.hiddenOut,
                    cellIn: classifierOutput.cellOut
                )
            } else {
                classifierInput = GestureClassifierInput(features: modelInput)
            }
            
            do {
                classifierOutput = try classifier.prediction(input: classifierInput)
            } catch {
                fatalError("Failed to predict")
            }
            
            DispatchQueue.main.async {
                let result = self.classifierOutput.activity
                let confidence = self.classifierOutput.activityProbability[result]!
                self.resultLabel.text = result
                self.confidenceLabel.text = String(format: "%.1f%%", confidence * 100)
            }
        }
    }
}

