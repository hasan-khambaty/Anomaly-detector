Anomalyze: Network Traffic Anomaly Detection Tool

Inspiration

We have always been intrigued by how hackers bypass systems and gain unauthorized access. This curiosity inspired us to explore how cybersecurity professionals detect and classify malicious activities. Anomalyze was born out of this pursuit—a tool designed to identify malicious activity and distinguish it from normal network traffic. This project allowed us to uncover patterns, distributions, and intricate details involved in tracking attackers.

What It Does

Anomalyze evaluates network connections using 13 critical features, such as source and destination IPs, packet lengths, frequencies, and protocols. The tool learns patterns in normal and malicious network traffic to classify them effectively. It prioritizes security by minimizing false negatives, ensuring that potentially harmful activity is not overlooked.

How We Built It

Data Collection:
Simulated attacks like Distributed Denial-of-Service (DDoS), Remote Access Trojan (RAT), Man-in-the-Middle (MITM), and IP spoofing.
Captured network traffic using Wireshark, generating over 300,000 entries with detailed connection information.
Model Development:
Trained a voting classifier combining LightGBM and Random Forest.
Fine-tuned the model using cross-validation, hyperparameter optimization, and metrics like accuracy, precision, recall, and F1 score.
Validation:
Tested the model against new simulated attacks to verify its robustness and sensitivity.
Enhanced adaptability by augmenting data and adjusting features with tools like SHAP and permutation importance.
Challenges

Feature Distributions: Training data captured specific distributions for features like packet length, while test data showed significant deviations. Data augmentation techniques helped mitigate these challenges.
Simulated vs. Real-World Scenarios: Simulated attacks lacked real-world variability, leading to false negatives. We retrained the model with improved variability to address this mismatch.
Key Insights

Protocols: Attackers often use a limited set of protocols to minimize detection.
IP Address Patterns: Malicious traffic often targets specific IP blocks, while normal traffic shows a wide range of IP activity.
Packet Length: Shorter packets were indicative of malicious traffic.
Port Numbers: Higher ports were more frequently associated with malicious activity.
QUIC_DCID_Encoded: Attackers exploit modern protocols with irregular identifiers to bypass detection mechanisms.
What's Next for Anomalyze

Probability Scores: Add probabilities to predictions for more flexible thresholds.
Reducing False Positives: Train with diverse datasets to improve real-world applicability.
Real-Time Explainability: Integrate feature-level explainability for informed decision-making.
Contextual Data: Incorporate geographical and temporal data to enhance detection.
Built With

Python
LightGBM
RandomForestClassifier
Matplotlib
Pandas
Seaborn
SHAP
Permutation-Importance
Installation

Step 1: Clone the Repository
Clone the GitHub repository to your local machine:

https://github.com/hasan-khambaty/Anomaly-detector/ 

cd Anomaly-detector  

Step 2: Set Up a Virtual Environment

Create a Python virtual environment:

python -m venv anomalyze_env  

Activate the virtual environment:

On Windows:

.\anomalyze_env\Scripts\activate  

On macOS/Linux:
source anomalyze_env/bin/activate  
Install the required packages:

pip install joblib lightgbm matplotlib numpy pandas seaborn scikit-learn streamlit 

Step 3: Run the Application

Start the Streamlit application:

streamlit run app.py  

This will open the application in your default web browser.

Creating Your Own CSV Files

If you want to create your own CSV files for analysis:

Capture Network Data:

Use Wireshark to capture network traffic.

Export the data as a CSV file using the "Make CSV" function in Wireshark.

Label the Data:

Use labeladder.py to add a column specifying whether the traffic is normal or malicious.

Merge Multiple CSVs:

Use mergecsv.py to combine all your labeled CSV files into a single dataset.

Analyze the Data:

Load the combined CSV file into the Anomalyze application and explore the visualizations and predictions.

Accomplishments

Minimized false negatives for high sensitivity to harmful activity.
Achieved high accuracy, precision, recall, and F1 score for robust performance.
Contributing

Contributions are welcome! Fork the repository and submit a pull request for enhancements or bug fixes.

License

This project is licensed under the MIT License.

Devpost

Check out our project on Devpost: [Anomalyze](https://devpost.com/software/anomalyze)
