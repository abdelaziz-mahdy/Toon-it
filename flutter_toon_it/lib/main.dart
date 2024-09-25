import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

void main() {
  runApp(const CartoonizeApp());
}

class CartoonizeApp extends StatelessWidget {
  const CartoonizeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Cartoonize App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const CartoonizeHomePage(),
    );
  }
}

class CartoonizeHomePage extends StatefulWidget {
  const CartoonizeHomePage({super.key});

  @override
  _CartoonizeHomePageState createState() => _CartoonizeHomePageState();
}

class _CartoonizeHomePageState extends State<CartoonizeHomePage> {
  Uint8List? _originalImage;
  Uint8List? _cartoonImage;
  List<Uint8List> _processSteps = [];

  final ImagePicker _picker = ImagePicker();
  bool _isProcessing = false;

  // Function to pick an image from the gallery
  Future<void> pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      setState(() {
        _originalImage = bytes;
        _cartoonImage = null; // Reset the cartoonized image
        _processSteps = [];
      });
      await applyCartoonize(bytes);
    }
  }

  // Function to apply the cartoonization process
  Future<void> applyCartoonize(Uint8List imageBytes) async {
    setState(() {
      _isProcessing = true;
    });

    // Decode the image to Mat format
    final img = cv.imdecode(imageBytes, cv.IMREAD_COLOR);

    // Step 1: Convert to Grayscale
    final gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    _addStepToProcess(gray, "Grayscale");

    // Step 2: Apply Gaussian Blur
    final blurred = cv.gaussianBlur(gray, (7, 7), 0);
    _addStepToProcess(blurred, "Blurred");

    // Step 3: Edge Detection using Laplacian
    final edges = cv.laplacian(blurred, cv.MatType.CV_8U);
    _addStepToProcess(edges, "Edges");

    // Step 4: Threshold the edges
    final (_, binaryEdges) = cv.threshold(edges, 80, 255, cv.THRESH_BINARY);
    _addStepToProcess(binaryEdges, "Binary Edges");

    // Step 5: Use bitwise AND to merge edges and original image
    final cartoonized = cv.bitwiseAND(img, img, mask: binaryEdges);
    final cartoonImageEncoded = cv.imencode(".png", cartoonized).$2;

    setState(() {
      _cartoonImage = cartoonImageEncoded;
      _isProcessing = false;
    });
  }

  // Helper function to add steps to the processSteps list
  void _addStepToProcess(cv.Mat mat, String stepName) {
    final (success, bytes) = cv.imencode(".png", mat);
    if (success) {
      setState(() {
        _processSteps.add(bytes);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cartoonize Image'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (_isProcessing) const LinearProgressIndicator(),
            const SizedBox(height: 20),
            _originalImage != null
                ? Image.memory(_originalImage!, height: 200)
                : const Text("No image selected"),
            const SizedBox(height: 20),
            _cartoonImage != null
                ? Image.memory(_cartoonImage!, height: 200)
                : const Text("Cartoonized image will appear here"),
            const SizedBox(height: 20),
            _buildProcessStepsView(),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: pickImage,
              child: const Text("Pick Image"),
            ),
          ],
        ),
      ),
    );
  }

  // Widget to display process steps
  Widget _buildProcessStepsView() {
    if (_processSteps.isEmpty) {
      return const Text("Processing steps will appear here.");
    }

    return SizedBox(
      height: 120,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: _processSteps.length,
        itemBuilder: (context, index) {
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
            child: Column(
              children: [
                Image.memory(_processSteps[index], width: 80, height: 80),
                Text("Step ${index + 1}"),
              ],
            ),
          );
        },
      ),
    );
  }
}
