# 🦜✨ Bird Snap – Your AI Birding Companion!  

## 🔍 **What is Bird Snap?**  
Ever spotted a bird and wondered what species it is? 🧐 Just snap a photo, and **Bird Snap** will do the rest! Powered by **deep learning**, this tool can identify over **500 bird species** trained on **NABirds** with an impressive **87% Top-1 and 96% Top-5 accuracy**! Whether you're a seasoned birder, a casual nature lover, or simply curious, Bird Snap brings the world of birds to your fingertips. 🌿🐦  

🎯 **Try it out now! → [Live Web App](https://birdsnap.streamlit.app/)**  

<!-- ![Bird Snap Demo](https://github.com/user-attachments/assets/)   -->

## 🌟 **Why You’ll Love Bird Snap**  
- 🧠 **AI-Powered Identification** – Uses **EfficientNetB4** to classify birds with high accuracy.  
- 🔥 **Fine-Tuned for Birds** – Trained on **50,000+ stunning bird photos** for top-tier precision.  
- 🖥 **Interactive Web App** – Simply upload a photo, and Bird Snap will name the species.  
- 📖 **Learn & Explore** – Instantly access Wikipedia info about each identified bird.  

## 🛠 **How Does It Work?**  

### 🖼 1. Smarter Data Augmentation  
Birds come in all shapes, colors, and lighting conditions. To handle this variety, Bird Snap uses advanced augmentation techniques like:  
✅ **Flipping**  
✅ **Zooming & Rotation**  
✅ **Contrast & Brightness Adjustments**  
This makes the model **robust** to different environments and bird postures.  

### 🚀 2. Transfer Learning with EfficientNet  
Bird Snap’s backbone is **EfficientNetB4**, a **state-of-the-art** deep learning model pre-trained on **ImageNet**. We fine-tuned it to:  
- Extract **bird-specific features** for classification.  
- Improve accuracy with custom layers designed for birds.  

### 🎯 3. Fine-Tuned for Perfection  
To maximize accuracy, the final **10 layers** were fine-tuned for an additional **5 epochs**, making Bird Snap even sharper at recognizing subtle species differences.  

## 📊 **Model Performance**  
Bird Snap was tested on unseen bird images and achieved:  
- **Accuracy**: 87%  
- **Precision**: 88%  
- **Recall**: 86%  
- **F1 Score**: 87%  
With results like these, you can trust Bird Snap to be your expert birding companion! 🐦✨  

## 🖥 **Interactive Web App Features**  
Bird Snap is more than just an AI model—it’s a **fully functional web app** built with **Streamlit**! Here’s what you can do:  
1. **Upload a Bird Photo** – Drag & drop your image for instant identification.  
2. **Get Probability Insights** – View the **top 3 predicted species** with confidence scores.  
3. **Learn More** – Each bird’s name links to its **Wikipedia page** for deeper exploration.  
4. **See Sample Images** – Compare your bird with similar species from the test dataset.  

## 🚀 **Get Started**  

### 📌 Prerequisites  
- **Python 3.8+**  
- Required libraries: **TensorFlow, Streamlit, NumPy, Pandas, etc.** (see `requirements.txt`).  

### 📦 Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/zysea23/bird-snap.git
   ```

2. Navigate to the project directory:
   ```bash
   cd bird-snap
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🐤 Why Birds?
Birds are nature’s storytellers, carrying mysteries in their songs, feathers, and migrations. With **over 10,000 species** worldwide, every bird is a discovery waiting to happen. This project is dedicated to making bird identification **fun, accessible, and educational** for all. 🌎💚

## 🤝 Contributions
Help Bird Snap soar to new heights! Contributions are welcome—whether it's improving the model, refining the web app, or expanding the dataset. 🦅

## 📜 License
This project is licensed under the MIT License. Feel free to use, modify, and share!

## 🌈 Acknowledgments
**EfficientNet** – The backbone of Bird Snap’s recognition power.
**Streamlit** – For making the web app smooth and intuitive.
**Wikipedia API** – For enriching the app with detailed bird info.

🚀 Let’s make birdwatching smarter, one snap at a time! Happy birding! 🐦❤️

