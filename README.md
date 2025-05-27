
## 🌍 Satellite Image to Map Conversion and Land Cover Analysis

This project leverages a **Pix2Pix GAN** architecture to convert satellite imagery into human-readable map visuals and performs **land cover analysis** to detect vegetation, water bodies, and land regions. The system is packaged into an intuitive **Streamlit** web application for real-time use.

---

## 🚀 Features

- 🔁 **Image-to-Image Translation** using Pix2Pix GAN (U-Net Generator + PatchGAN Discriminator)
- 🗺️ **Map Generation** from real satellite images
- 🌿 **Land Cover Analysis** to compute percentage of vegetation, water, and land
- ⚙️ **Streamlit UI** for interactive image upload, map generation, and result visualization
- 🧠 **Trained on paired datasets** (Satellite ↔ Map) for accurate geospatial translation

---

## 🛠️ Tech Stack

- **Deep Learning Framework**: PyTorch  
- **Frontend**: Streamlit  
- **Image Processing**: OpenCV, Albumentations  
- **Visualization**: Matplotlib, Pillow  
- **GAN Type**: Pix2Pix (Conditional GAN)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/satellite-map-gan.git
cd satellite-map-gan
pip install -r requirements.txt
````

> ✅ Ensure your system has a CUDA-compatible GPU for faster processing.

---

## 🖼️ Usage

### 🏋️‍♂️ Train the model (or load a pre-trained checkpoint):

```bash
python train.py
```

### 🚀 Launch the Streamlit app:

```bash
streamlit run main.py
```

### 📥 Upload a satellite image and get:

* A generated map image
* Land cover analysis with percentage of vegetation, water, and land
* Highlighted vegetation and water features

---

## 🧪 Testing

* ✅ **Unit Tests** for preprocessing, model inference, and land classification
* 🔁 **Integration Tests** ensure smooth flow from upload → generation → analysis
* 🧑‍💻 **User Testing** confirms the UI is simple and intuitive

---

## 📊 Results

* 🎯 High-quality, visually accurate **map outputs** from diverse satellite images
* 🌱 Reliable **land/vegetation/water segmentation**
* ⚡ Real-time performance with **Streamlit interface**

---

## 🔮 Future Enhancements

* 🔢 Support for **multi-class segmentation** (buildings, roads, barren land, etc.)
* 🌐 **GIS integration** for professional applications
* ⚡ **Real-time inference** with TensorRT/ONNX
* 🗺️ Extended **dataset coverage** for various terrains and seasons

---

## 📸 Sample Output

> *Add sample before-and-after images of satellite input and generated map here*

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

