import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('combined.csv')

normal_entries = df[df['label'] == 'normal']
malicious_entries = df[df['label'] == 'malicious']

sampled_normal = normal_entries.sample(n=50000, random_state=42, replace=len(normal_entries) < 50000)
sampled_malicious = malicious_entries.sample(n=50000, random_state=42, replace=True)

balanced_df = pd.concat([sampled_normal, sampled_malicious], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_df.index += 1

if 'No.' not in balanced_df.columns:
    balanced_df.rename_axis('No.', inplace=True)
    balanced_df.reset_index(inplace=True)

balanced_df.to_csv('balanced_combined.csv', index=False)

df2 = pd.read_csv('balanced_combined.csv')
df2 = df2.drop('No.', axis=1)
df2['label'] = df2['label'].apply(lambda x: 0 if x == 'normal' else 1)

def parse_info(row):
    protocol = row['Protocol']
    info = row['Info']

    if protocol in ['TCP', 'UDP']:
        ports = re.search(r'(\d+)\s*>\s*(\d+)', info)
        flags = re.search(r'\[(.*?)\]', info)
        seq = re.search(r'Seq=(\d+)', info)
        ack = re.search(r'Ack=(\d+)', info)
        return {
            'Source_Port': ports.group(1) if ports else '-1',
            'Destination_Port': ports.group(2) if ports else '-1',
            'Flags': flags.group(1) if flags else 'None',
            'Sequence': seq.group(1) if seq else 0,
            'Acknowledgment': ack.group(1) if ack else 0,
            'Other_Info': 'None'
        }
    elif protocol in ['ICMP', 'ICMPv6']:
        message = re.search(r'(Echo.*?request|Echo.*?reply)', info)
        return {'ICMP_Type': message.group(1) if message else 'None', 'Other_Info': 'None'}
    elif protocol in ['QUIC']:
        dcid = re.search(r'DCID=([a-zA-Z0-9]+)', info)
        return {'QUIC_DCID': dcid.group(1) if dcid else 'Unknown', 'Other_Info': 'None'}
    elif protocol in ['DNS', 'MDNS']:
        query = re.search(r'Standard query ([^ ]+)', info)
        return {'DNS_Query_Type': query.group(1) if query else 'None', 'Other_Info': 'None'}
    else:
        return {'Other_Info': info}

parsed_info = df2.apply(parse_info, axis=1)
parsed_df = pd.DataFrame(parsed_info.tolist())
df2 = pd.concat([df2, parsed_df], axis=1)

df2['Source_Port'] = df2['Source_Port'].fillna('-1')
df2['Destination_Port'] = df2['Destination_Port'].fillna('-1')
df2['Source_Port'] = df2['Source_Port'].replace('-1', 'None')
df2['Destination_Port'] = df2['Destination_Port'].replace('-1', 'None')

df2 = df2.drop('Info', axis=1)
df2 = df2.drop('Other_Info', axis=1)

def categorize_flags(flag):
    if pd.isna(flag):
        return 'No_Flags'
    elif flag in ['SYN', 'SYN, ACK']:
        return 'Connection_Setup'
    elif flag in ['FIN, ACK', 'RST', 'RST, ACK']:
        return 'Connection_Termination'
    elif flag in ['PSH, ACK', 'ACK']:
        return 'Data_Transfer'
    elif flag.startswith('TCP'):
        return 'TCP_Diagnostics'
    else:
        return 'Other'
df2['Flag_Category'] = df2['Flags'].apply(categorize_flags)

#label encoding flags
category_mapping = {
    'No_Flags': 0,
    'Connection_Setup': 1,
    'Connection_Termination': 2,
    'Data_Transfer': 3,
    'TCP_Diagnostics': 4,
    'Other': 5
}
df2['Flag_Category'] = df2['Flag_Category'].map(category_mapping)

#dropping og flag column
df2 = df2.drop('Flags', axis=1)

#dropping columns that have over 90% nan and dont
#explain sig variation of the target
df2 = df2.drop(['DNS_Query_Type'], axis=1)
df2 = df2.drop(['ICMP_Type'], axis=1)

#if quic_dcid has high freq it is normal
#if quic_dcid has low freq is can be potentially malicious so we create
#a new column denoting respecting freq of each quic_dcid entry so that model can take into account
#frequency also whichll help it in predicting label

#high freq means normal repeated connections
#attackers avoid reuing connection to stay undetected so thats why low freq usually sign of problem

#so we frequency encode we dont one hot encode as this would create like 1000 more columns and label encoding doesnt rlly
#provide any conbtext to the problem

df2['QUIC_DCID'] = df2['QUIC_DCID'].fillna('Unknown')
dcid_counts = df2['QUIC_DCID'].value_counts()
df2['DCID_Frequency'] = df2['QUIC_DCID'].map(dcid_counts)


#udp is unordered communication so seq has no value but in wireshark sometimes
#seq column gets filled with vals so -1 to ensure model doesnt misinterpret the data
#and affect preds

df2.loc[(df2['Protocol'] == 'UDP') & (df2['Sequence'] == 0), 'Sequence'] = -1
df2['Sequence'] = df2['Sequence'].fillna(-1)

df2['Acknowledgment'] = df2['Acknowledgment'].fillna(-1)

#source/dest port contains the actual port num(tells about type of communication)
#source/dest encoded contains the ip addresses which are categorical data so they are label encoded to numeric(tells
#about source or dest of sender/reciever)

df2['Source_Port'] = df2['Source_Port'].replace('None', np.nan).fillna(-1)
df2['Destination_Port'] = df2['Destination_Port'].replace('None', np.nan).fillna(-1)

ip_encoder = LabelEncoder()
df2['Source_Encoded'] = ip_encoder.fit_transform(df2['Source'])
df2['Destination_Encoded'] = ip_encoder.fit_transform(df2['Destination'])

quic_dcid_encoder = LabelEncoder()
df2['QUIC_DCID'] = df2['QUIC_DCID'].replace('unknown', 'unknown_value')
df2['QUIC_DCID_Encoded'] = quic_dcid_encoder.fit_transform(df2['QUIC_DCID'])

protocol_encoder = LabelEncoder()
df2['Protocol_Encoded'] = protocol_encoder.fit_transform(df2['Protocol'])

columns_to_fix = ['Source_Port', 'Destination_Port', 'Sequence', 'Acknowledgment']
for col in columns_to_fix:
    df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(-1)

columns_to_drop = ['Source', 'Destination', 'QUIC_DCID', 'Protocol']
df2 = df2.drop([col for col in columns_to_drop if col in df2.columns], axis=1)

def add_synthetic_variability(X, noise_level=0.08):
    X_augmented = X.copy()

    for col in X_augmented.select_dtypes(include=[np.number]).columns:
        noise = np.random.normal(0, noise_level * X_augmented[col].std(), X_augmented.shape[0])
        X_augmented[col] += noise

    for col in X_augmented.select_dtypes(include=[object, 'category']).columns:
        X_augmented[col] = X_augmented[col].sample(frac=1).reset_index(drop=True)

    return X_augmented

X = df2.drop(['label', 'Time'], axis=1)
y = df2['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.4)

X_train_augmented = add_synthetic_variability(X_train)
X_train_combined = pd.concat([X_train, X_train_augmented], axis=0)
y_train_combined = pd.concat([y_train, y_train], axis=0)

lgb_model = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.1, random_state=42)
rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)

model2 = VotingClassifier(
    estimators=[('lgb', lgb_model), ('rf', rf_model)],
    voting='hard'
)

model2.fit(X_train_combined, y_train_combined)
y_pred2 = model2.predict(X_test)

joblib.dump(model2, 'voting_classifier_model.pkl')

#evaluation

#acc of predictions
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Accuracy: {accuracy2}")
#acc = tp + tn/total samples

#acc of train data to check for potential overfitting
train_accuracy2 = accuracy_score(y_train, model2.predict(X_train))
print(f"Training Accuracy: {train_accuracy2}")

#precision
precision = precision_score(y_test, y_pred2)
print(f"Precision: {precision}")
#true pos/true pos + false pos

#recall
recall = recall_score(y_test, y_pred2)
print(f"Recall: {recall}")
#true pos/true pos+false negatives


#f1 score
f1 = f1_score(y_test, y_pred2)
print(f"F1-Score: {f1}")
#harmonic mean of precision and recall
# f1 score = 2 *(precision * recall)/(precision+recall)

#conf matrix
cm = confusion_matrix(y_test, y_pred2)
print(f"Confusion Matrix:\n{cm}")

#reading new csv w test dataset
st.title("Anomaly Detection App")

csv_file = st.selectbox("Select CSV File:", ["combined.csv", "combined_output1.csv"])
predict_df = pd.read_csv(csv_file)
st.write(f"### Random Sample of 5353 Entries from Uploaded Dataset:")
st.write(predict_df.sample(5353))

predict_df['label'] = predict_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
predict_df = predict_df.dropna(subset=['label'])

predict_df = predict_df.sample(n=5353, random_state=42)

#preprocessing data and handling missing vals
parsed_info_predict = predict_df.apply(parse_info, axis=1)
parsed_df_predict = pd.DataFrame(parsed_info_predict.tolist())
predict_df = pd.concat([predict_df, parsed_df_predict], axis=1)

predict_df['Source_Port'] = predict_df['Source_Port'].fillna('-1').replace('-1', 'None')
predict_df['Destination_Port'] = predict_df['Destination_Port'].fillna('-1').replace('-1', 'None')
predict_df = predict_df.drop(['Info', 'Other_Info'], axis=1)
predict_df['Flag_Category'] = predict_df['Flags'].apply(categorize_flags).map(category_mapping)
predict_df = predict_df.drop('Flags', axis=1)


if 'QUIC_DCID' in predict_df.columns:
    predict_df['QUIC_DCID'] = predict_df['QUIC_DCID'].fillna('Unknown')
    dcid_counts_predict = predict_df['QUIC_DCID'].value_counts()
    predict_df['DCID_Frequency'] = predict_df['QUIC_DCID'].map(dcid_counts_predict)

if 'Protocol' in predict_df.columns:
    predict_df['Protocol'] = predict_df['Protocol'].apply(
        lambda x: x if x in protocol_encoder.classes_ else 'Unknown'
    )
    if 'Unknown' not in protocol_encoder.classes_:
        protocol_encoder.classes_ = np.append(protocol_encoder.classes_, 'Unknown')
    predict_df['Protocol_Encoded'] = protocol_encoder.transform(predict_df['Protocol'])


predict_df['Source'] = predict_df['Source'].apply(lambda x: x if x in ip_encoder.classes_ else 'unknown_value')
predict_df['Destination'] = predict_df['Destination'].apply(lambda x: x if x in ip_encoder.classes_ else 'unknown_value')
ip_encoder.classes_ = np.append(ip_encoder.classes_, 'unknown_value')
predict_df['Source_Encoded'] = ip_encoder.transform(predict_df['Source'])
predict_df['Destination_Encoded'] = ip_encoder.transform(predict_df['Destination'])

for col in columns_to_fix:
    predict_df[col] = pd.to_numeric(predict_df[col], errors='coerce').fillna(-1)

predict_df = predict_df.drop(['Time', 'No.'], axis=1, errors='ignore')
predict_df = predict_df.dropna(subset=['label'])
true_labels2 = predict_df['label']


raw_pred_vals = predict_df.drop(['label'], axis=1)

missing_cols = [col for col in X_train.columns if col not in raw_pred_vals.columns]
for col in missing_cols:
    raw_pred_vals[col] = -1


raw_pred_vals = raw_pred_vals[X_train.columns]
model_raw_pred2 = model2.predict(raw_pred_vals)


if len(model_raw_pred2) != len(true_labels2):
    raise ValueError(f"Length mismatch: Predictions ({len(model_raw_pred2)}) vs True Labels ({len(true_labels2)})")

visualization = st.selectbox("Select Visualization:", ["Evaluation Metrics" , "Confusion Matrix" , "True vs Predicted Labels Distribution"
                            , "Feature Importance Chart" , "Class Distribution" , "Missing Data Visualizer"])

labels = ['Normal (0)', 'Malicious (1)']

if visualization == "Evaluation Metrics":
    st.write("### Evaluation Metrics:")
    normal_count = (predict_df['label'] == 0).sum()
    malicious_count = (predict_df['label'] == 1).sum()
    st.write(f"Sampled Dataset: {normal_count} Normal, {malicious_count} Malicious")

    accuracy = accuracy_score(true_labels2, model_raw_pred2)
    st.write(f"\nOverall Accuracy: {accuracy}")

    overall_f1 = f1_score(true_labels2, model_raw_pred2)
    st.write(f"Overall F1-Score: {overall_f1}")

    overall_precision = precision_score(true_labels2, model_raw_pred2)
    st.write(f"Overall Precision: {overall_precision}")

    overall_recall = recall_score(true_labels2, model_raw_pred2)
    st.write(f"Overall Recall: {overall_recall}")

    overall_auc_roc = roc_auc_score(true_labels2, model_raw_pred2)
    st.write(f"Overall AUC-ROC: {overall_auc_roc}")

#conf matrix
elif visualization == "Confusion Matrix":
    st.write("### Confusion Matrix:")
    conf_matrix = confusion_matrix(true_labels2, model_raw_pred2)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted Labels', fontsize=14)
    ax.set_ylabel('True Labels', fontsize=14)
    st.pyplot(fig)


#true vs pred labels distribution chart
elif visualization == "True vs Predicted Labels Distribution":
    st.write("### True vs Predicted Labels Distribution:")
    true_counts = true_labels2.value_counts(sort=False)
    predicted_counts = pd.Series(model_raw_pred2).value_counts(sort=False)


    true_counts_aligned = [true_counts.get(0, 0), true_counts.get(1, 0)]
    predicted_counts_aligned = [predicted_counts.get(0, 0), predicted_counts.get(1, 0)]
    bar_width = 0.4
    indices = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.4
    indices = range(len(labels))
    ax.bar(indices, true_counts_aligned, width=bar_width, label='True Labels', color='blue', alpha=0.7)
    ax.bar([i + bar_width for i in indices], predicted_counts_aligned, width=bar_width, label='Predicted Labels', color='orange', alpha=0.7)
    ax.set_xticks([i + bar_width / 2 for i in indices])
    ax.set_xticklabels(labels)
    ax.set_title('True vs Predicted Labels Distribution', fontsize=16)
    ax.set_xlabel('Labels', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.legend()
    st.pyplot(fig)


    #feature importance chart
elif visualization == "Feature Importance Chart":
    st.write("### Feature Importance Chart:")
    data_with_labels = pd.concat([raw_pred_vals, true_labels2.rename("label")], axis=1)

    correlation_features = [col for col in data_with_labels.columns if col in X_train.columns or col == "label"]
    correlation_matrix = data_with_labels[correlation_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix[["label"]].drop("label"),  # Focus only on correlations with the label
        annot=True, cmap="coolwarm", fmt=".2f"
    )
    plt.title("Correlation of Features with Target Label")
    plt.show()

    lgb_importances = model2.named_estimators_['lgb'].feature_importances_
    rf_importances = model2.named_estimators_['rf'].feature_importances_
    combined_importances = np.mean([lgb_importances, rf_importances], axis=0)
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": combined_importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importances (Averaged from LightGBM and Random Forest)", fontsize=16)
    ax.set_xlabel("Importance Score", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    st.pyplot(fig)

    aligned_features = list(set(correlation_matrix.index).intersection(importance_df["Feature"]))

elif visualization == "Class Distribution":
    # Class Distribution
    st.write("### Class Distribution:")
    class_counts = y.value_counts()
    fig, ax = plt.subplots()
    ax.bar(class_counts.index, class_counts.values, color=['blue', 'orange'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal (0)', 'Malicious (1)'])
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    st.pyplot(fig)

    #Missing Data Visualizer
elif visualization == "Missing Data Visualizer":
        st.write("### Missing Data Visualization:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(predict_df.isnull(), cbar=False, cmap='coolwarm', ax=ax)
        ax.set_title('Missing Data Heatmap')
        st.pyplot(fig)

st.subheader("Key Takeaways:")
st.write(f"We observed that protocols emerged as one of the most critical factors in predicting malicious activity, as attackers rarely employ a diverse set of protocols, relying instead on a few well-known ones that support their malicious operations. This aligns with the theory that attackers prioritize stealth and efficiency, often choosing protocols that enable rapid exploitation while minimizing their chances of detection. For example, malicious traffic often avoids standard secure protocols like HTTPS in favor of less-regulated ones.")

st.write("Higher source IP addresses (Source_Encoded) were generally associated with normal traffic, while higher destination IP addresses (Destination_Encoded) were also linked to normal activity. This supports the theory that normal network behavior typically involves a wide range of IPs at both source and destination ends, such as connections from dynamic IP pools of legitimate users to cloud servers or service endpoints. Malicious traffic, on the other hand, tends to concentrate on specific IP blocks or use low-range, less-dynamic IP addresses, as attackers often leverage compromised devices or fixed systems for their operations.")

st.write("Packet length showed a significant relationship with the target label, with shorter packets being more indicative of malicious traffic. This aligns with known attack patterns where attackers send minimal data (e.g., ping floods or request-based attacks) to probe systems or exploit vulnerabilities while avoiding detection by large data transfers that might raise suspicion.")

st.write("Port numbers also provided key insights, with higher port numbers frequently associated with malicious activity. This observation fits theoretical expectations, as lower ports are typically reserved for system or application-level processes, such as HTTP (port 80) or SMTP (port 25), which are less likely to be misused directly in attacks. Conversely, higher ports are often used in malicious operations to disguise traffic or communicate with compromised systems, especially during attacks like botnet communications.")

st.write("QUIC_DCID_Encoded emerged as another highly significant feature. This likely reflects the fact that attackers often use unusual or irregular identifiers in their QUIC-based connections to bypass traditional inspection mechanisms. This aligns with the theory that attackers exploit the flexibility of modern protocols to introduce variability, making their activity harder to classify based on predefined rules.")

st.write("These findings reinforce the theoretical understanding of malicious behavior in networks: attackers exploit predictable behaviors (e.g., using limited protocols, higher ports, or fixed identifiers) to achieve their objectives while trying to evade detection mechanisms. By leveraging these theoretical foundations, the model successfully prioritizes features that distinguish normal from malicious traffic.")