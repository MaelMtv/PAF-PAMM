import SwiftUI
import UIKit

let customColor = Color(hue: 0.63, saturation: 0.75, brightness: 0.31)

struct ContentView: View {
    @State private var isImagePickerPresented = false
    @State private var isCameraPickerPresented = false
    @State private var selectedImage: UIImage?
    @State private var isCheckButtonVisible = false
    @State private var brisqueScore: String?
    @State private var diseaseResult: String? // Add this line

    var body: some View {
        ZStack {
            VStack {
                HStack {
                    Spacer()
                    Button(action: {
                        selectedImage = nil
                        isCheckButtonVisible = false
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .padding()
                            .foregroundColor(.white)
                    }
                }
                Spacer()
                Text("DermaGuard")
                    .font(.largeTitle)
                    .fontWeight(.heavy)
                    .foregroundColor(Color.white)
                    .multilineTextAlignment(.center)
                    
                Text("Bienvenue dans l'application de diagnostic des maladies de la peau")
                    .font(.subheadline)
                    .fontWeight(.light)
                    .foregroundColor(Color.white)
                    .multilineTextAlignment(.center)
                    .padding()
                
                if let capturedImage = selectedImage {
                    Image(uiImage: capturedImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 400, height: 400)
                    
                    if let score = brisqueScore {
                        Text("Score BRISQUE : \(score)")
                            .fontWeight(.semibold)
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                    }

                    // Display disease result
                    if let disease = diseaseResult {
                        Text("Resultat: \(disease)")
                            .fontWeight(.semibold)
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                    }
                    
                    if isCheckButtonVisible {
                        HStack {
                            Button(action: {
                                self.sendImage(image: capturedImage)
                            }) {
                                HStack {
                                    Image(systemName: "checkmark")
                                        .font(.headline)
                                    Text("Score")
                                        .fontWeight(.semibold)
                                        .font(.headline)
                                }
                            }
                            .buttonStyle(GradientBackgroundStyle())

                            Button(action: {
                                // Here you should call your machine learning model or function
                                if let capturedImage = selectedImage {
                                    self.sendImageForDiseaseCheck(image: capturedImage)
                                }
                            }) {
                                HStack {
                                    Image(systemName: "arrow.triangle.2.circlepath")
                                        .font(.headline)
                                    Text("Prediction")
                                        .fontWeight(.semibold)
                                        .font(.headline)
                                }
                            }
                            .buttonStyle(GradientBackgroundStyle())
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                Spacer()
                
                VStack {
                    Button(action: {
                        isImagePickerPresented = true
                    }) {
                        HStack {
                            Image(systemName: "photo.on.rectangle")
                                .font(.headline)
                            Text("Importer")
                                .fontWeight(.semibold)
                                .font(.headline)
                        }
                    }
                    .buttonStyle(GradientBackgroundStyle())
                    
                    Button(action: {
                        isCameraPickerPresented = true
                    }) {
                        HStack {
                            Image(systemName: "camera")
                                .font(.headline)
                            Text("Camera")
                                .fontWeight(.semibold)
                                .font(.headline)
                        }
                    }
                    .buttonStyle(GradientBackgroundStyle())
                }
                .frame(maxWidth: .infinity, alignment: .bottomTrailing)
                .padding(.bottom, 50)
            }
            .background(customColor.edgesIgnoringSafeArea(.all))
        }
        .sheet(isPresented: $isImagePickerPresented, onDismiss: {
            if let selectedImage = selectedImage {
                self.selectedImage = selectedImage
                isCheckButtonVisible = true
            }
        }) {
            ImagePickerView(sourceType: .photoLibrary, selectedImage: $selectedImage)
        }
        .sheet(isPresented: $isCameraPickerPresented, onDismiss: {
            if let selectedImage = selectedImage {
                self.selectedImage = selectedImage
                isCheckButtonVisible = true
            }
        }) {
            ImagePickerView(sourceType: .camera, selectedImage: $selectedImage)
        }
    }


    func sendImage(image: UIImage) {
        guard let imageData = image.pngData() else {
            print("Failed to get PNG data from image")
            return
        }

        let imageBase64 = imageData.base64EncodedString()

        let url = URL(string: "http://medelbechir.pythonanywhere.com/brisque")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = try? JSONEncoder().encode(["image": imageBase64])
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error: \(error)")
            } else if let data = data {
                if let decodedResponse = try? JSONDecoder().decode(BrisqueResponse.self, from: data) {
                    DispatchQueue.main.async {
                        self.brisqueScore = String(decodedResponse.brisque)
                    }
                } else {
                    print("Failed to decode response")
                }
            } else {
                print("No data received")
            }
        }.resume()
    }

    // Add this function
    func sendImageForDiseaseCheck(image: UIImage) {
        guard let imageData = image.jpegData(compressionQuality: 0.9) else {
            print("Failed to get JPEG data from image")
            return
        }

        let imageBase64 = imageData.base64EncodedString()

        let url = URL(string: "http://mmebechir.pythonanywhere.com/predict")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = try? JSONEncoder().encode(["file": imageBase64])
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error: \(error)")
            } else if let data = data {
                do {
                    let decodedResponse = try JSONDecoder().decode(DiseaseResponse.self, from: data)
                    DispatchQueue.main.async {
                        self.diseaseResult = decodedResponse.result
                    }
                } catch {
                    print("Failed to decode JSON response: \(error)")
                    if let response = response as? HTTPURLResponse {
                        print("HTTP status code: \(response.statusCode)")
                    }
                    if let json = String(data: data, encoding: .utf8) {
                        print("JSON response: \(json)")
                    }
                }
            } else {
                print("No data received")
            }
        }.resume()
    }


}

struct BrisqueResponse: Codable {
    let brisque: Float
}

// Add this struct
struct DiseaseResponse: Codable {
    let result: String
}


struct GradientBackgroundStyle: ButtonStyle {
    func makeBody(configuration: Self.Configuration) -> some View {
        configuration.label
            .frame(minWidth: 0, maxWidth: .infinity)
            .padding()
            .foregroundColor(.white)
            .background(Color.black)
            .cornerRadius(30)
            .padding(.horizontal, 40)
            .scaleEffect(configuration.isPressed ? 0.9 : 1.0)
    }
}

struct ImagePickerView: UIViewControllerRepresentable {
    let sourceType: UIImagePickerController.SourceType
    @Binding var selectedImage: UIImage?

    func makeCoordinator() -> Coordinator {
        Coordinator(selectedImage: $selectedImage)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // Update the view controller if needed
    }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        @Binding var selectedImage: UIImage?

        init(selectedImage: Binding<UIImage?>) {
            _selectedImage = selectedImage
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                selectedImage = image
            }

            picker.dismiss(animated: true, completion: nil)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true, completion: nil)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
