import pandas as pd
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

################ Q1 ################

################ Task A ################

url = 'https://raw.githubusercontent.com/aalanwar/Logical-Zonotope/refs/heads/main/README.md'
response = urllib.request.urlopen(url)
text = response.read().decode('utf-8')

# Keep only letters and replace everything else with space
clean_text = ''
for ch in text.lower():
    if 'a' <= ch <= 'z':
        clean_text += ch
    else:
        clean_text += ' '


#split the words and adding them in the dictionary
words = clean_text.split()
word_counts = {}
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1


################ Task B ################


# Define stopwords - unwanted words
stopwords = {
    'the', 'a', 'an', 'be', 'to', 'of', 'in', 'on', 'is', 'and', 'it', 'for', 'with', 'by', 'that', 'as', 'are',
    'br', 'div', 'span', 'href', 'html', 'head', 'body', 'title', 'meta', 'http', 'https', 'www', 'img', 'src'
}

# Create a new dictionary with stopwords removed
filtered_counts = {}

for word, count in word_counts.items():
    if word in stopwords:
        continue
    if len(word) < 2:  # skip very short junk words
        continue
    filtered_counts[word] = count

word_series = pd.Series(filtered_counts)

# Sort by values (counts) descending and take top 10
top_words = word_series.sort_values(ascending=False).head(10)


#plotting the top 10 words

top_words.plot(kind='bar', color='skyblue', figsize=(10,6))
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



################ Task C ################


n, m = 100, 20
A = np.random.rand(n, m) 

# Create vector v with normal distribution (mean=2, std=0.01)
mu, sigma = 2, 0.01
v = np.random.normal(loc=mu, scale=sigma, size=(20, 1))  

# Display shapes to be sure
print(f"A shape: {A.shape}")
print(f"v shape: {v.shape}")
# Calculate c
c = (A * v.T).sum(axis=0).reshape(20, 1)
#calculate STD and MEAN
mean_c = np.mean(c)
std_c = np.std(c)
#display histogram of C
plt.hist(c.flatten(), bins=5, edgecolor='black')
plt.title('Histogram of Vector c')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

################ Q2 ################

### Q2 dot 2 ####
def learn_simple_linreg(x, y):
    """this function learns the parameters of a simple linear regression model
    given the data x and y."""
    x = x.flatten() 
    y = y.flatten()
    
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    numerator = np.sum((x - x_bar) * (y - y_bar))
    denominator = np.sum((x - x_bar) ** 2)
    
    beta_1 = numerator / denominator
    beta_0 = y_bar - beta_1 * x_bar
    
    return beta_0, beta_1
### Q2 dot 3 ####

def predict_simple_linreg(x, beta_0, beta_1):
    """this function predicts the output of a simple linear regression model
    given the input x and the learned parameters beta_0 and beta_1."""
    return beta_0 + beta_1 * x

### Q2 dot 1 ####

mu = 2
sigmas = [0.01, 0.1, 1]
n_rows, n_cols = 100, 2

# Generate 3 datasets with increasing standard deviations
A_1 = np.random.normal(loc=mu, scale=sigmas[0], size=(n_rows, n_cols))
A_2 = np.random.normal(loc=mu, scale=sigmas[1], size=(n_rows, n_cols))
A_3 = np.random.normal(loc=mu, scale=sigmas[2], size=(n_rows, n_cols))



datasets = [A_1, A_2, A_3]
titles = ['A_1 (σ=0.01)', 'A_2 (σ=0.1)', 'A_3 (σ=1)']

plt.figure(figsize=(15, 4))

for i, A in enumerate(datasets):
    x_train = A[:, 0]
    y_train = A[:, 1]
    beta_0, beta_1 = learn_simple_linreg(x_train, y_train)
    
    x_line = np.linspace(np.min(x_train), np.max(x_train), 100)
    y_line = predict_simple_linreg(x_line, beta_0, beta_1)

    plt.subplot(1, 3, i+1)
    plt.scatter(x_train, y_train, color='blue', alpha=0.6, label='Training Data')
    plt.plot(x_line, y_line, color='red', label='Regression Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titles[i])
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

### Q2 dot 4 ####
def learn_simple_linreg_with_intercept(x, y):
    """this function learns the parameters of a simple linear regression model
    given the data x and y."""
    x = x.flatten() 
    y = y.flatten()
    
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    numerator = np.sum((x - x_bar) * (y - y_bar))
    denominator = np.sum((x - x_bar) ** 2)
    
    beta_1 = numerator / denominator
    beta_0 = y_bar - beta_1 * x_bar
    
    return beta_0, beta_1



### Q2 dot 5 ####

"""its seems that the higher the sigma, the line get less and less positive """

### Q2 dot 6 ###

for i, A in enumerate([A_1, A_2, A_3]):
    x_train = A[:, 0]
    y_train = A[:, 1]

    _, beta_1 = learn_simple_linreg(x_train, y_train)
    beta_0 = 0  # Force β₀ to zero

    x_line = np.linspace(np.min(x_train), np.max(x_train), 100)
    y_line = predict_simple_linreg(x_line, beta_0, beta_1)

    plt.subplot(1, 3, i + 1)
    plt.scatter(x_train, y_train, color='blue', alpha=0.6, label='Training Data')
    plt.plot(x_line, y_line, color='red', label='β₀ = 0 Line')
    plt.title(f'A_{i+1}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

""" the conclustion is that the Beta_0 made the model much more sensitive to the data and to the MU"""
### Q2 dot 7 ###

plt.figure(figsize=(15, 4))

for i, A in enumerate(datasets):
    x_train = A[:, 0]
    y_train = A[:, 1]

    beta_0, _ = learn_simple_linreg(x_train, y_train)
    beta_1 = 0  # Force slope to zero

    x_line = np.linspace(np.min(x_train), np.max(x_train), 100)
    y_line = predict_simple_linreg(x_line, beta_0, beta_1)

    plt.subplot(1, 3, i + 1)
    plt.scatter(x_train, y_train, color='blue', alpha=0.6, label='Training Data')
    plt.plot(x_line, y_line, color='red', label='Regression Line (β₁ = 0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titles[i])
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

""" As expected we get a straight line with the value of beta_0 as the mean 
of the data.The model ignores x entirely and predicts the same constant value for all inputs — the (fixed) intercept."""


### Q2 dot 8 ###
def learn_simple_linreg_2(x, y):
    x = x.flatten()
    y = y.flatten()

    # Construct design matrix X with a column of 1s for the intercept (β₀)
    X = np.column_stack((np.ones_like(x), x))  # shape: (N, 2)

    # Solve for [β₀, β₁] using least squares
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    beta_0, beta_1 = beta
    return beta_0, beta_1


#### Q2 dot 9 ###
from sklearn.linear_model import LinearRegression

def learn_simple_linreg_3(x, y):
    x = x.reshape(-1, 1)  # Make x a 2D column vector as required by sklearn
    y = y.flatten()       # Ensure y is 1D

    model = LinearRegression()
    model.fit(x, y)

    beta_0 = model.intercept_
    beta_1 = model.coef_[0]
    return beta_0, beta_1
### Q2 dot 9- Example usage ###
datasets = [A_1, A_2, A_3]
titles = ['A_1 (σ=0.01)', 'A_2 (σ=0.1)', 'A_3 (σ=1)']

print("Comparing β₀ and β₁ from np.linalg.lstsq vs sklearn.linear_model.LinearRegression\n")
print(f"{'Dataset':<15} {'β₀_lstsq':>10} {'β₁_lstsq':>10} {'β₀_sklearn':>12} {'β₁_sklearn':>12}")

for i, A in enumerate(datasets):
    x = A[:, 0]
    y = A[:, 1]

    # lstsq version
    b0_lstsq, b1_lstsq = learn_simple_linreg_2(x, y)

    # sklearn version
    b0_sklearn, b1_sklearn = learn_simple_linreg_3(x, y)

    print(f"{titles[i]:<15} {b0_lstsq:>10.4f} {b1_lstsq:>10.4f} {b0_sklearn:>12.4f} {b1_sklearn:>12.4f}")
