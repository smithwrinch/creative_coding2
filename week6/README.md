# Week 6 - Introduction to Machine Learning II
This week we looked more deeply into neural network concepts using jupyter notebooks. I was tasked to add a new transformation to the supplied dataset of faces.
## My implementation
I decided to investigate Principle Component Analysis. This is a dimensionaility reduction technique that allows data to be analysed much more efficiently. This is really useful for images since they contain width*height features which is far too many for most machine learning methods to work with. It finds an amount of principle components (= no data points supplies) and you can select the top X to fit your confidence level. To ensure I had a thorough understanding of this technique I applied it automatically using scikit-learn and manually using numpy. I also decided to convert the images to grayscale as this reduces information by a third.

### Automatic implentation using scikit-learn
This illustrates the top 12 (arbitrary) eigenfaces.
```
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


grayscale = np.zeros(data.shape[:-1])
h = grayscale.shape[2]
w = grayscale.shape[1]
for i in range(grayscale.shape[0]):
    grayscale[i] = rgb2gray(data[i])

# d = grayscale[1:].ravel()
d = grayscale.transpose(1,2,0).reshape(-1, grayscale.shape[0])
d = d.transpose(1,0)
# d = d.ravel()
n_components = 12
# print(d2.shape)
pca = PCA(n_components=n_components, whiten=True).fit(d)
eigenfaces = pca.components_.reshape((n_components, w, h))


# apply PCA transformation (to use if I were to train a network)
data_pca = pca.transform(d)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
plot_gallery(eigenfaces, range(n_components), w, h)

```
![automatic](img/automatic.png)
### Manual principle component analysis

```
standardised_data = np.zeros(grayscale.shape)
mean_img_gr = rgb2gray(mean_img)
std_img_gr = rgb2gray(std_img)
for i in range(grayscale.shape[0]):
    standardised_data[i] = (grayscale[i] - mean_img_gr) / std_img_gr
```
Firstly I standardise the data. After converting to grayscale, this involves subtracting the mean image and dividing by the standard deviation.
\
![std](img/std.png)
```
# formatting to 2D array
M = standardised_data.transpose(1,2,0).reshape(-1, standardised_data.shape[0])
covariance_matrix = np.cov(M.T)
```

Now I reshape the data such that I can find the covariance matrix as this requires a 2D matrix. The covariance matrix can be seen as a measure of how much each dimension varies from the mean compared to one another. That is, it contains information about the amount of variance shared between pairs of dimensions.
\
\
![std](img/cov.png)
```
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) #eigenfaces are the eigenvectors
```
The eigenvectors are the principle components, this makes sense as eigen vectors are orthogonal to each another so provide the basis for the data points. Thankfully I didn't have to revisit any computational linear algebra from university and I could find the eigendecomposition of the covariance matrix using numpy's `linalg`.  
```
# To work out the PCA transformation matrix

# Calculating the explained variance on each of components
variance_explained = []
for i in eigen_values:
     variance_explained.append((i/sum(eigen_values))*100)

# print(variance_explained)
# 18.7% of variance i ndata is explained by first principle component
# Identifying components that explain at least 80%
cumulative_variance_explained = np.cumsum(variance_explained)
# This tells me I need the first 10 components for 80% confidence level

plt.plot(range(cumulative_variance_explained.shape[0]), cumulative_variance_explained)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained variance vs Number of components")
plt.show()

# The components whose eigenvalue falls above the point where the slope of the
# line changes most drastically is called the "elbow"

```
Here I am trying to find the optimal amount of principle components to be used for analysis. Due to the small dataset, I opted for a 80% confidence level which came to about 20 components
\
![std](img/variance.png)
```
# PCA transformation matrix using first 20 principle components
projection_matrix = (eigen_vectors.T[:][:20]).T
data_pca2 = M.dot(projection_matrix)


eigenfaces2 = data_pca2.T.reshape((20, w, h))
for i in range(20):
    eigenfaces2[i] = (eigenfaces2[i]  * std_img_gr) + mean_img_gr

plot_gallery(eigenfaces2, range(20), w, h, 4, 5)
```

\
![std](img/automatic.png)
