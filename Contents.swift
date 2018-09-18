import Foundation

//
//  Heap.swift
//  Written for the Swift Algorithm Club by Kevin Randrup and Matthijs Hollemans
//
public struct Heap<T> {
    
    /** The array that stores the heap's nodes. */
    var nodes = [T]()
    
    /**
     * Determines how to compare two nodes in the heap.
     * Use '>' for a max-heap or '<' for a min-heap,
     * or provide a comparing method if the heap is made
     * of custom elements, for example tuples.
     */
    private var orderCriteria: (T, T) -> Bool
    
    /**
     * Creates an empty heap.
     * The sort function determines whether this is a min-heap or max-heap.
     * For comparable data types, > makes a max-heap, < makes a min-heap.
     */
    public init(sort: @escaping (T, T) -> Bool) {
        self.orderCriteria = sort
    }
    
    /**
     * Creates a heap from an array. The order of the array does not matter;
     * the elements are inserted into the heap in the order determined by the
     * sort function. For comparable data types, '>' makes a max-heap,
     * '<' makes a min-heap.
     */
    public init(array: [T], sort: @escaping (T, T) -> Bool) {
        self.orderCriteria = sort
        configureHeap(from: array)
    }
    
    /**
     * Configures the max-heap or min-heap from an array, in a bottom-up manner.
     * Performance: This runs pretty much in O(n).
     */
    private mutating func configureHeap(from array: [T]) {
        nodes = array
        for i in stride(from: (nodes.count/2-1), through: 0, by: -1) {
            shiftDown(i)
        }
    }
    
    public var isEmpty: Bool {
        return nodes.isEmpty
    }
    
    public var count: Int {
        return nodes.count
    }
    
    /**
     * Returns the index of the parent of the element at index i.
     * The element at index 0 is the root of the tree and has no parent.
     */
    @inline(__always) internal func parentIndex(ofIndex i: Int) -> Int {
        return (i - 1) / 2
    }
    
    /**
     * Returns the index of the left child of the element at index i.
     * Note that this index can be greater than the heap size, in which case
     * there is no left child.
     */
    @inline(__always) internal func leftChildIndex(ofIndex i: Int) -> Int {
        return 2*i + 1
    }
    
    /**
     * Returns the index of the right child of the element at index i.
     * Note that this index can be greater than the heap size, in which case
     * there is no right child.
     */
    @inline(__always) internal func rightChildIndex(ofIndex i: Int) -> Int {
        return 2*i + 2
    }
    
    /**
     * Returns the maximum value in the heap (for a max-heap) or the minimum
     * value (for a min-heap).
     */
    public func peek() -> T? {
        return nodes.first
    }
    
    /**
     * Adds a new value to the heap. This reorders the heap so that the max-heap
     * or min-heap property still holds. Performance: O(log n).
     */
    public mutating func insert(_ value: T) {
        nodes.append(value)
        shiftUp(nodes.count - 1)
    }
    
    /**
     * Adds a sequence of values to the heap. This reorders the heap so that
     * the max-heap or min-heap property still holds. Performance: O(log n).
     */
    public mutating func insert<S: Sequence>(_ sequence: S) where S.Iterator.Element == T {
        for value in sequence {
            insert(value)
        }
    }
    
    /**
     * Allows you to change an element. This reorders the heap so that
     * the max-heap or min-heap property still holds.
     */
    public mutating func replace(index i: Int, value: T) {
        guard i < nodes.count else { return }
        
        remove(at: i)
        insert(value)
    }
    
    /**
     * Removes the root node from the heap. For a max-heap, this is the maximum
     * value; for a min-heap it is the minimum value. Performance: O(log n).
     */
    @discardableResult public mutating func remove() -> T? {
        guard !nodes.isEmpty else { return nil }
        
        if nodes.count == 1 {
            return nodes.removeLast()
        } else {
            // Use the last node to replace the first one, then fix the heap by
            // shifting this new first node into its proper position.
            let value = nodes[0]
            nodes[0] = nodes.removeLast()
            shiftDown(0)
            return value
        }
    }
    
    /**
     * Removes an arbitrary node from the heap. Performance: O(log n).
     * Note that you need to know the node's index.
     */
    @discardableResult public mutating func remove(at index: Int) -> T? {
        guard index < nodes.count else { return nil }
        
        let size = nodes.count - 1
        if index != size {
            nodes.swapAt(index, size)
            shiftDown(from: index, until: size)
            shiftUp(index)
        }
        return nodes.removeLast()
    }
    
    /**
     * Takes a child node and looks at its parents; if a parent is not larger
     * (max-heap) or not smaller (min-heap) than the child, we exchange them.
     */
    internal mutating func shiftUp(_ index: Int) {
        var childIndex = index
        let child = nodes[childIndex]
        var parentIndex = self.parentIndex(ofIndex: childIndex)
        
        while childIndex > 0 && orderCriteria(child, nodes[parentIndex]) {
            nodes[childIndex] = nodes[parentIndex]
            childIndex = parentIndex
            parentIndex = self.parentIndex(ofIndex: childIndex)
        }
        
        nodes[childIndex] = child
    }
    
    /**
     * Looks at a parent node and makes sure it is still larger (max-heap) or
     * smaller (min-heap) than its childeren.
     */
    internal mutating func shiftDown(from index: Int, until endIndex: Int) {
        let leftChildIndex = self.leftChildIndex(ofIndex: index)
        let rightChildIndex = leftChildIndex + 1
        
        // Figure out which comes first if we order them by the sort function:
        // the parent, the left child, or the right child. If the parent comes
        // first, we're done. If not, that element is out-of-place and we make
        // it "float down" the tree until the heap property is restored.
        var first = index
        if leftChildIndex < endIndex && orderCriteria(nodes[leftChildIndex], nodes[first]) {
            first = leftChildIndex
        }
        if rightChildIndex < endIndex && orderCriteria(nodes[rightChildIndex], nodes[first]) {
            first = rightChildIndex
        }
        if first == index { return }
        
        nodes.swapAt(index, first)
        shiftDown(from: first, until: endIndex)
    }
    
    internal mutating func shiftDown(_ index: Int) {
        shiftDown(from: index, until: nodes.count)
    }
    
}

// MARK: - Searching
extension Heap where T: Equatable {
    
    /** Get the index of a node in the heap. Performance: O(n). */
    public func index(of node: T) -> Int? {
        return nodes.index(where: { $0 == node })
    }
    
    /** Removes the first occurrence of a node from the heap. Performance: O(n log n). */
    @discardableResult public mutating func remove(node: T) -> T? {
        if let index = index(of: node) {
            return remove(at: index)
        }
        return nil
    }
    
}

let stopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"]
var stopWordsSet = Set<String>()
for word in stopWords {
    stopWordsSet.insert(word)
}

enum DataClass: String {
    case PlusOne = "+1"
    case MinusOne = "-1"
}


var memoIdf: [String : Float] = [:]

public class Document {
    
    var dataClass: DataClass?
    var frequencies: [String : (Int)] = [:]
    let review: String
    var reviewWordCount: Int = 0
    var tfidf: [String: Float] = [:]
    var predictedClass: DataClass?
    
    init(str: String, test: Bool = false) {
        var index = str.startIndex
        if(!test) {
            index = str.index(str.startIndex, offsetBy: 2)
            let strDataClass = String(str[..<index])
            self.dataClass = DataClass(rawValue: strDataClass)!
        }
        
        var strReview = String(str[index ..< str.endIndex]).lowercased()
        self.review = strReview
        
        let tagger = NSLinguisticTagger(tagSchemes: [.lemma], options: 0)
        let options: NSLinguisticTagger.Options = [.omitPunctuation, .omitWhitespace]
        tagger.string = strReview
        let range = NSRange(location: 0, length: strReview.utf16.count)
        
        var strReviewfrequencies: [String : Int] = [:]
        
        tagger.enumerateTags(in: range, unit: .word, scheme: .lemma, options: options) { tag, tokenRange, stop in
            
            if let lemma = tag?.rawValue.lowercased() {
//                if stopWordsSet.contains(lemma) { return }
                self.reviewWordCount += 1
                if strReviewfrequencies[lemma] == nil {
                    strReviewfrequencies[lemma] = 1
                } else {
                    strReviewfrequencies[lemma] = strReviewfrequencies[lemma]! + 1
                }
            }
        }
        self.frequencies = strReviewfrequencies
    }
    
    func tfNorm(word: String) -> Float? {
        return Float(self.frequencies[word]!) / Float(self.reviewWordCount)
    }
    
    func idf(word: String, documents: [Document]) -> Float {
        if memoIdf.keys.contains(word) {
            return memoIdf[word]!
        }
        var hits: Float = 0.0
        for doc in documents {
            if doc.frequencies.keys.contains(word) {
                hits += 1.0
            }
        }
        let finalIdf = log(Float(documents.count) / hits)
        memoIdf[word] = finalIdf
        return finalIdf
    }
}

func kSorted(array: [(Float, Document)], k: Int) -> [(Float, Document)] {
    var array = array
    // Insert first K + 1 items in min heap
    var heap = Heap<(Float, Document)> { (one, two) -> Bool in
        return one.0 < two.0
    }
    for i in 0..<k {
        heap.insert(array[i])
    }
    var index = 0
    for i in (k+1)..<array.count {
        array[index] = heap.peek()!
        index += 1
        heap.remove()
        heap.insert(array[i])
    }
    while(!heap.isEmpty) {
        array[index] = heap.peek()!
        index += 1
        heap.remove()
    }
    return array
}

var currentDirectoryPath = FileManager.default.currentDirectoryPath
currentDirectoryPath.append("/Resources/train.data")

let data = try! String(contentsOfFile: currentDirectoryPath, encoding: String.Encoding.ascii)
var linedData = data.components(separatedBy: "\n")
if linedData.last!.isEmpty { linedData.removeLast() }

//var counter = 1
var plusOneCounter = 0
var minusOneCounter = 0
var documents: [Document] = []
for lineData in linedData {

//    print(counter)
    let document = Document(str: lineData)
//    if document.dataClass == .MinusOne {
//        minusOneCounter += 1
//    } else {
//        plusOneCounter += 1
//    }
    documents.append(document)
}
//    counter += 1

//print("PlusOneCounter: \(plusOneCounter)")
//print("MinusOneCounter: \(minusOneCounter)")
//exit(0)
//counter = 1
print("♦️♦️♦️♦️♦️♦️♦️")
for doc in documents {
//    print(counter)
    for (word,_) in doc.frequencies {
        let tfNorm: Float = doc.tfNorm(word: word)!
        let idf: Float = doc.idf(word: word, documents: documents)

        let tfidf = tfNorm * idf
        doc.tfidf[word] = tfidf
    }
//    counter += 1
}
print("♦️♦️♦️♦️♦️♦️♦️")

func kNearestNeighbor(k: Int, documents:[Document], unclassifiedDocument: Document) -> DataClass {
    var cosineSimilarity: [(Float,Document)] = []
    // Do the cosine similarity algo here and append each value to list
    
    for (word,_) in unclassifiedDocument.frequencies {
        let tfNorm: Float = unclassifiedDocument.tfNorm(word: word)!
        let idf: Float = unclassifiedDocument.idf(word: word, documents: documents)
        
        let tfidf = tfNorm * idf
        unclassifiedDocument.tfidf[word] = tfidf
    }
    
    for doc in documents {
        var dotProduct: Float = 0.0
        var unclassifiedDocumentNormalized: Float = 0.0 // Still needs to be squarerooted
        var docNormalized: Float = 0.0 // Still needs to be squarerooted
        for (word, tfidf) in unclassifiedDocument.tfidf {
            if doc.tfidf[word] != nil {
                dotProduct += tfidf * doc.tfidf[word]!
                docNormalized += doc.tfidf[word]! * doc.tfidf[word]!
            }
            unclassifiedDocumentNormalized += tfidf * tfidf
        }
        let cosineSimilarityValue = dotProduct / (sqrtf(docNormalized) * sqrtf(unclassifiedDocumentNormalized))
        cosineSimilarity.append((cosineSimilarityValue, doc))
    }
//    cosineSimilarity = kSorted(array: cosineSimilarity, k: k)
    cosineSimilarity.sort { (floatDoc1, floatDoc2) -> Bool in
        return floatDoc1.0 > floatDoc2.0
    }
    var plusOneCount = 0
    var minusOneCount = 0
    for i in 0..<k {
        let document = cosineSimilarity[i].1
        if document.dataClass == .PlusOne {
            plusOneCount += 1
        } else {
            minusOneCount += 1
        }
    }
    unclassifiedDocument.predictedClass = plusOneCount >= minusOneCount ? .PlusOne : .MinusOne
//    print("Prediction: \(unclassifiedDocument.dataClass == unclassifiedDocument.predictedClass!)")
    return unclassifiedDocument.predictedClass!
}

print("Testing Phase 💚💚💚💚💚")
// Now begin the testing phase
var testPath = FileManager.default.currentDirectoryPath
testPath.append("/Resources/finaltest.data")

let testData = try! String(contentsOfFile: testPath, encoding: String.Encoding.ascii)
var testLinedData = testData.components(separatedBy: "\n")

var testDocuments: [Document] = []
for linedData in testLinedData {
    let document = Document(str: linedData, test: true)
    testDocuments.append(document)
}
print("Classifying Phase 🧡🧡🧡🧡🧡🧡")

var counter = 0
var outputString = ""
for doc in testDocuments {
    print(counter)
    let output = kNearestNeighbor(k:2021, documents: documents, unclassifiedDocument: doc)
    outputString += "\(output.rawValue)\n"
    counter += 1
}

try! outputString.write(toFile: FileManager.default.currentDirectoryPath + "/results.txt", atomically: false, encoding: .utf16)
print(outputString)
