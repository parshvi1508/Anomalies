# E-Learning Anomaly Detection Dashboard 📊

A Streamlit-based dashboard for analyzing student learning patterns, detecting anomalies, and predicting dropout risks in e-learning environments.

## Features 🌟

- **Data Analysis & Visualization**
  - Course engagement tracking
  - Test score distribution
  - Time spent analysis
  - Location/device switching patterns
  - Correlation analysis

- **Anomaly Detection**
  - Multiple anomaly type detection
  - Customizable filtering options
  - Real-time data filtering
  - Interactive visualizations

- **Risk Analysis**
  - Dropout risk prediction
  - Automated recommendations
  - Student performance tracking
  - Engagement level monitoring

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anomaly_detection.git
cd anomaly_detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Required Dependencies 📦

- streamlit
- pandas
- seaborn
- matplotlib
- plotly
- scikit-learn
- numpy
- joblib

## Usage 🚀

1. Start the Streamlit app:
```bash
streamlit run app2.py
```

2. Upload your student data CSV file containing the following required columns:
   - student_id
   - video_completion_rate
   - quiz_accuracy
   - avg_time_per_video
   - forum_activity
   - location_change
   - num_course_views

3. Use the sidebar filters to analyze specific student segments

## Data Format 📄

The input CSV should follow this structure:
```csv
student_id,video_completion_rate,quiz_accuracy,avg_time_per_video,forum_activity,location_change,num_course_views
1,75,82,25,3,2,150
2,45,35,15,1,4,80
...
```

## Features Explanation 🔍

### Filters
- **Quiz Score Range**: Filter students by their quiz performance
- **Engagement Level**: Filter by Low/Medium/High engagement
- **Forum Activity**: Filter by minimum forum participation
- **Anomaly Types**: Filter specific patterns of concern

### Visualizations
- Course view distribution
- Time spent analysis
- Device switching patterns
- Score distribution
- Correlation analysis

### Risk Analysis
- ML-based dropout prediction
- Personalized recommendations
- Risk factor breakdown
- Automated intervention suggestions

## Contributing 🤝

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 

- Built with Streamlit
- Uses scikit-learn for ML components
- Plotly for interactive visualizations