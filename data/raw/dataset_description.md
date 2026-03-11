# Drought Prediction Dataset Description

## 1. Overview
本数据集用于研究加州 Fresno 地区干旱条件下植被的动态响应。数据融合了 NASA 卫星遥感数据与 NOAA 地面气象观测数据。

## 2. Data Sources
- **Vegetation Data (NDVI):** - Source: NASA LP DAAC (AppEEARS)
  - Product: MOD13Q1 (v006)
  - Frequency: 16-day composite
  - Resolution: 250m
- **Meteorological Data:**
  - Source: NOAA NCEI (Climate Data Online)
  - Station: FRESNO YOSEMITE INTERNATIONAL AIRPORT, CA US
  - Frequency: Daily

## 3. Data Dictionary
| Column Name | Description | Unit/Range | Source |
|-------------|-------------|------------|--------|
| Date        | Observation date | YYYY-MM-DD | Both |
| NDVI        | Normalized Difference Vegetation Index | 0.0 to 1.0 | NASA |
| Precipitation| Daily total precipitation | Inches | NOAA |
| Temp        | Average daily air temperature | Fahrenheit | NOAA |

## 4. Preprocessing
- **Alignment:** Meteorological data and NDVI data are merged by date.
- **Interpolation:** Since NDVI is available every 16 days, linear interpolation is used to generate daily values for time-series modeling.
- **Normalization:** All features are scaled to [0, 1] using Min-Max scaling.