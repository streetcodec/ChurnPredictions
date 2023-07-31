from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("customer_data.csv")


# Using 'age' and 'total_spend' for segmentation
X_segmentation = data[['age', 'total_spend']]

# Train the K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
data['segment'] = kmeans.fit_predict(X_segmentation)


plt.scatter(data['age'], data['total_spend'], c=data['segment'], cmap='rainbow')
plt.xlabel("Age")
plt.ylabel("Total Spend")
plt.title("Customer Segmentation")
plt.show()
