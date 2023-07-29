from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load data
data = pd.read_csv("customer_data.csv")

# Step 2: Preprocess data
# Assuming we want to use 'age' and 'total_spend' for segmentation
X_segmentation = data[['age', 'total_spend']]

# Step 3: Train the K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
data['segment'] = kmeans.fit_predict(X_segmentation)

# Step 4: Visualize customer segmentation
plt.scatter(data['age'], data['total_spend'], c=data['segment'], cmap='rainbow')
plt.xlabel("Age")
plt.ylabel("Total Spend")
plt.title("Customer Segmentation")
plt.show()
