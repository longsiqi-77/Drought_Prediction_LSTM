cat <<EOF > README.md
# DeepDrought: Multi-source Data Fusion for Drought Prediction

## 🚀 Overview
**DeepDrought** is a deep learning system designed to predict vegetation drought trends by fusing **ground-level meteorological data** (NOAA) and **satellite remote sensing data** (MODIS NDVI). It specifically addresses the **lag effect** of vegetation response to climate changes.

## 🛠 Key Features
- **Data Fusion**: Automated alignment of daily weather data and 16-day NDVI imagery via linear interpolation.
- **Architectures**: Comparison between **Bi-LSTM** (temporal features) and **Transformer** (long-range dependencies).
- **Engineering**: Integrated with **WandB** for experiment tracking and **Streamlit** for an interactive visualization dashboard.

## 📊 Performance
- **Transformer R²**: 0.8145
- **LSTM R²**: 0.7239
- *The Transformer model demonstrates superior robustness in multi-step forecasting.*

## 📂 Structure
- \`src/\`: Modular source code (Models, DataLoader, Trainer)
- \`dashboard/\`: Streamlit UI implementation
- \`results/\`: Prediction figures and metrics

---
**Course**: Big Data Systems | **Dev Date**: Jan 2025
EOF
