# NumPy Quick Reference Guide

A comprehensive cheat sheet for NumPy operations and functions.

## Table of Contents
- [Installation & Import](#installation--import)
- [Array Creation](#array-creation)
- [Array Properties](#array-properties)
- [Data Types](#data-types)
- [Indexing & Slicing](#indexing--slicing)
- [Reshaping](#reshaping)
- [Basic Operations](#basic-operations)
- [Broadcasting](#broadcasting)
- [Linear Algebra](#linear-algebra)
- [Random Numbers](#random-numbers)
- [Sorting & Searching](#sorting--searching)
- [Advanced Indexing](#advanced-indexing)
- [Array Manipulation](#array-manipulation)
- [Set Operations](#set-operations)
- [File I/O](#file-io)
- [Universal Functions](#universal-functions)
- [Statistics](#statistics)
- [Polynomials](#polynomials)
- [Structured Arrays](#structured-arrays)
- [Performance Tips](#performance-tips)

---

## Installation & Import

```python
# Install
pip install numpy

# Import
import numpy as np

# Check version
np.__version__
```

---

## Array Creation

```python
# From lists
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])

# Zeros, ones, empty
np.zeros((3, 4))           # 3x4 array of zeros
np.ones((2, 3))            # 2x3 array of ones
np.empty((2, 2))           # Uninitialized 2x2 array
np.full((3, 3), 7)         # 3x3 array filled with 7

# Identity matrix
np.eye(4)                  # 4x4 identity matrix
np.identity(3)             # 3x3 identity matrix

# Ranges
np.arange(10)              # 0 to 9
np.arange(5, 15)           # 5 to 14
np.arange(0, 20, 2)        # 0 to 18, step 2
np.linspace(0, 1, 5)       # 5 evenly spaced values from 0 to 1
np.logspace(0, 3, 4)       # 10^0 to 10^3, 4 values

# From existing arrays
np.zeros_like(arr)         # Zeros with same shape as arr
np.ones_like(arr)          # Ones with same shape as arr
np.copy(arr)               # Deep copy of arr
```

---

## Array Properties

```python
arr.shape                  # Dimensions (rows, cols)
arr.ndim                   # Number of dimensions
arr.size                   # Total number of elements
arr.dtype                  # Data type of elements
arr.itemsize               # Size of each element in bytes
arr.nbytes                 # Total bytes consumed
arr.T                      # Transpose
```

---

## Data Types

```python
# Integer types
np.int8, np.int16, np.int32, np.int64
np.uint8, np.uint16, np.uint32, np.uint64

# Float types
np.float16, np.float32, np.float64

# Other types
np.bool_                   # Boolean
np.complex64, np.complex128  # Complex numbers

# Create with specific dtype
np.array([1, 2, 3], dtype=np.float32)

# Convert dtype
arr.astype(int)
arr.astype(np.float64)
```

---

## Indexing & Slicing

```python
# 1D indexing
arr[0]                     # First element
arr[-1]                    # Last element
arr[2:5]                   # Elements 2, 3, 4
arr[:3]                    # First 3 elements
arr[::2]                   # Every 2nd element
arr[::-1]                  # Reverse array

# 2D indexing
arr[1, 2]                  # Row 1, column 2
arr[0]                     # First row
arr[:, 0]                  # First column
arr[0:2, 1:3]              # Subarray

# Boolean indexing
arr[arr > 5]               # Elements greater than 5
arr[(arr > 5) & (arr < 10)]  # Between 5 and 10

# Fancy indexing
arr[[0, 2, 4]]             # Elements at indices 0, 2, 4
```

---

## Reshaping

```python
arr.reshape(3, 4)          # Reshape to 3x4
arr.reshape(-1, 2)         # Auto-calculate rows, 2 columns
arr.flatten()              # Flatten to 1D (copy)
arr.ravel()                # Flatten to 1D (view if possible)
arr.T                      # Transpose
arr.transpose()            # Transpose

# Add/remove dimensions
arr[np.newaxis, :]         # Add dimension (row)
arr[:, np.newaxis]         # Add dimension (column)
arr.squeeze()              # Remove single-dimensional entries

# Resize
arr.resize((3, 4))         # Resize in-place
```

---

## Basic Operations

```python
# Arithmetic (element-wise)
arr + 5                    # Add scalar
arr * 2                    # Multiply by scalar
arr1 + arr2                # Add arrays
arr1 - arr2                # Subtract
arr1 * arr2                # Multiply
arr1 / arr2                # Divide
arr ** 2                   # Power

# Comparison
arr > 5                    # Boolean array
arr == arr2                # Element-wise equality

# Aggregations
arr.sum()                  # Sum all elements
arr.min()                  # Minimum
arr.max()                  # Maximum
arr.mean()                 # Average
arr.std()                  # Standard deviation
arr.var()                  # Variance
arr.median()               # Median

# With axis
arr.sum(axis=0)            # Sum down columns
arr.sum(axis=1)            # Sum across rows
arr.mean(axis=0)           # Mean of each column

# Positions
arr.argmin()               # Index of minimum
arr.argmax()               # Index of maximum
```

---

## Broadcasting

```python
# Scalar with array
arr + 10                   # Add 10 to all elements

# 1D with 2D
arr_2d + arr_1d            # Add 1D to each row

# Column broadcasting
arr_2d + arr_col[:, np.newaxis]

# Rules:
# 1. Arrays with fewer dimensions are padded with 1s
# 2. Arrays with size 1 in a dimension are stretched
# 3. Incompatible shapes raise ValueError
```

---

## Linear Algebra

```python
# Matrix multiplication
np.dot(A, B)               # Dot product
A @ B                      # Matrix multiplication (Python 3.5+)

# Matrix operations
np.linalg.inv(A)           # Inverse
np.linalg.det(A)           # Determinant
np.linalg.eig(A)           # Eigenvalues and eigenvectors
np.linalg.solve(A, b)      # Solve Ax = b
np.trace(A)                # Trace (sum of diagonal)
np.linalg.matrix_rank(A)   # Matrix rank
np.linalg.norm(A)          # Frobenius norm

# Decompositions
np.linalg.qr(A)            # QR decomposition
np.linalg.svd(A)           # Singular value decomposition
np.linalg.cholesky(A)      # Cholesky decomposition
```

---

## Random Numbers

```python
# Seed for reproducibility
np.random.seed(42)

# Random arrays
np.random.rand(3, 4)       # Uniform [0, 1)
np.random.randn(3, 4)      # Standard normal
np.random.randint(0, 10, size=(3, 4))  # Random integers

# Distributions
np.random.uniform(0, 10, size=5)       # Uniform distribution
np.random.normal(100, 15, size=5)      # Normal (mean, std)
np.random.binomial(10, 0.5, size=5)    # Binomial
np.random.poisson(3, size=5)           # Poisson

# Sampling
np.random.choice([1,2,3,4,5], size=3)  # Random choice
np.random.shuffle(arr)                 # Shuffle in-place
np.random.permutation(arr)             # Shuffled copy
```

---

## Sorting & Searching

```python
# Sorting
np.sort(arr)               # Return sorted copy
arr.sort()                 # Sort in-place
np.sort(arr)[::-1]         # Descending order
np.argsort(arr)            # Indices that would sort array

# 2D sorting
np.sort(arr, axis=0)       # Sort each column
np.sort(arr, axis=1)       # Sort each row

# Searching
np.where(arr > 5)          # Indices where condition is True
np.argmax(arr > 5)         # First index where True
np.searchsorted(arr, 5)    # Index to insert 5 in sorted array

# Checking
np.any(arr > 5)            # True if any element > 5
np.all(arr > 0)            # True if all elements > 0
```

---

## Advanced Indexing

```python
# np.where - conditional selection
np.where(arr > 5, arr, 0)  # Keep if >5, else 0
np.where(arr > 5, 'high', 'low')  # Conditional values

# np.select - multiple conditions
conditions = [arr < 5, arr < 10, arr >= 10]
choices = ['low', 'medium', 'high']
np.select(conditions, choices)

# np.choose - pick from arrays
indices = np.array([0, 1, 2, 0])
choices = [[10,10,10,10], [20,20,20,20], [30,30,30,30]]
np.choose(indices, choices)
```

---

## Array Manipulation

```python
# Concatenate
np.concatenate([arr1, arr2])           # Join 1D arrays
np.concatenate([arr1, arr2], axis=0)   # Vertical stack
np.concatenate([arr1, arr2], axis=1)   # Horizontal stack

# Stack
np.vstack([arr1, arr2])    # Vertical stack (rows)
np.hstack([arr1, arr2])    # Horizontal stack (columns)
np.dstack([arr1, arr2])    # Depth stack (3D)
np.column_stack([arr1, arr2])  # Stack as columns
np.row_stack([arr1, arr2])     # Stack as rows

# Split
np.split(arr, 3)           # Split into 3 equal parts
np.split(arr, [3, 7])      # Split at indices 3 and 7
np.vsplit(arr, 2)          # Split vertically
np.hsplit(arr, 2)          # Split horizontally

# Repeat and tile
np.repeat(arr, 3)          # Repeat each element 3 times
np.tile(arr, 3)            # Repeat entire array 3 times
np.tile(arr, (2, 3))       # Tile in 2D

# Insert and delete
np.insert(arr, 2, 99)      # Insert 99 at index 2
np.delete(arr, 2)          # Delete element at index 2
np.append(arr, [7, 8, 9])  # Append elements
```

---

## Set Operations

```python
# Unique values
np.unique(arr)                         # Unique values
np.unique(arr, return_counts=True)     # With counts
np.unique(arr, return_index=True)      # With first indices
np.unique(arr, return_inverse=True)    # With inverse indices

# Set operations
np.union1d(arr1, arr2)         # Union
np.intersect1d(arr1, arr2)     # Intersection
np.setdiff1d(arr1, arr2)       # Difference (in arr1, not arr2)
np.setxor1d(arr1, arr2)        # Symmetric difference
np.in1d(arr1, arr2)            # Element-wise membership test
```

---

## File I/O

```python
# Binary files (fast, NumPy-specific)
np.save('array.npy', arr)              # Save single array
arr = np.load('array.npy')             # Load array

np.savez('arrays.npz', a=arr1, b=arr2) # Save multiple arrays
data = np.load('arrays.npz')           # Load multiple
arr1 = data['a']

np.savez_compressed('data.npz', arr)   # Compressed save

# Text files (human-readable)
np.savetxt('data.txt', arr)            # Save as text
np.savetxt('data.csv', arr, delimiter=',')  # Save as CSV
arr = np.loadtxt('data.txt')           # Load from text
arr = np.loadtxt('data.csv', delimiter=',')  # Load CSV

# Advanced loading
np.genfromtxt('data.csv', delimiter=',', skip_header=1)
```

---

## Universal Functions

```python
# Mathematical
np.sqrt(arr)               # Square root
np.exp(arr)                # Exponential
np.log(arr)                # Natural log
np.log10(arr)              # Log base 10
np.log2(arr)               # Log base 2
np.power(arr, 3)           # Power
np.abs(arr)                # Absolute value
np.sign(arr)               # Sign (-1, 0, 1)

# Trigonometric
np.sin(arr)                # Sine
np.cos(arr)                # Cosine
np.tan(arr)                # Tangent
np.arcsin(arr)             # Arcsine
np.arccos(arr)             # Arccosine
np.arctan(arr)             # Arctangent
np.deg2rad(arr)            # Degrees to radians
np.rad2deg(arr)            # Radians to degrees

# Rounding
np.round(arr)              # Round to nearest
np.floor(arr)              # Round down
np.ceil(arr)               # Round up
np.trunc(arr)              # Truncate decimals
np.clip(arr, 0, 10)        # Clip values to range

# Comparison
np.maximum(arr1, arr2)     # Element-wise maximum
np.minimum(arr1, arr2)     # Element-wise minimum
np.greater(arr1, arr2)     # Element-wise >
np.less(arr1, arr2)        # Element-wise <

# Logical
np.logical_and(arr1, arr2) # Element-wise AND
np.logical_or(arr1, arr2)  # Element-wise OR
np.logical_not(arr)        # Element-wise NOT
np.logical_xor(arr1, arr2) # Element-wise XOR
```

---

## Statistics

```python
# Basic statistics
np.mean(arr)               # Mean
np.median(arr)             # Median
np.std(arr)                # Standard deviation
np.var(arr)                # Variance
np.min(arr)                # Minimum
np.max(arr)                # Maximum
np.sum(arr)                # Sum
np.prod(arr)               # Product

# Percentiles
np.percentile(arr, 25)     # 25th percentile
np.percentile(arr, [25, 50, 75])  # Multiple percentiles
np.quantile(arr, 0.5)      # Median (50th percentile)

# Correlation and covariance
np.corrcoef(arr1, arr2)    # Correlation coefficient
np.cov(arr1, arr2)         # Covariance

# Histogram
np.histogram(arr, bins=10) # Histogram counts and edges
np.digitize(arr, bins)     # Bin indices for each value

# Cumulative
np.cumsum(arr)             # Cumulative sum
np.cumprod(arr)            # Cumulative product
np.diff(arr)               # Differences between consecutive elements
```

---

## Polynomials

```python
# Create polynomial
p = np.poly1d([2, 3, 1])   # 2x² + 3x + 1

# Evaluate
p(5)                       # Evaluate at x=5
np.polyval([2, 3, 1], 5)   # Alternative

# Roots
np.roots([2, 3, 1])        # Find roots

# Fitting
np.polyfit(x, y, 2)        # Fit 2nd degree polynomial

# Calculus
np.polyder(p)              # Derivative
np.polyint(p)              # Integral

# Numerical
np.trapz(y, x)             # Trapezoidal integration
np.gradient(y, x)          # Numerical gradient
```

---

## Structured Arrays

```python
# Define dtype
dt = np.dtype([('name', 'U20'), ('age', 'i4'), ('score', 'f8')])

# Create structured array
arr = np.array([
    ('Alice', 25, 85.5),
    ('Bob', 30, 92.0)
], dtype=dt)

# Access fields
arr['name']                # All names
arr[0]['age']              # Alice's age

# Record arrays (easier access)
rec = np.rec.array(arr)
rec.name                   # Access as attribute
rec.score.mean()           # Statistics on field
```

---

## Performance Tips

```python
# Views vs Copies
view = arr[1:4]            # View (shares memory)
copy = arr[1:4].copy()     # Copy (independent)

# Check if view
arr.base is not None       # True if view

# Vectorization (ALWAYS prefer this)
result = arr ** 2 + 2*arr + 1  # Fast
# NOT: for loop                # Slow

# Pre-allocate
arr = np.zeros(1000)       # Pre-allocate
# NOT: np.append in loop     # Very slow

# In-place operations
arr *= 2                   # In-place (saves memory)
# arr = arr * 2             # Creates new array

# Memory order
arr_c = np.array(data, order='C')  # Row-major (default)
arr_f = np.array(data, order='F')  # Column-major

# Use appropriate dtype
arr = np.array(data, dtype=np.int32)  # Save memory if possible
```

---

## Common Patterns

```python
# Normalize data (0 to 1)
normalized = (arr - arr.min()) / (arr.max() - arr.min())

# Standardize (mean=0, std=1)
standardized = (arr - arr.mean()) / arr.std()

# Moving average
window = 5
moving_avg = np.convolve(arr, np.ones(window)/window, mode='valid')

# Distance matrix
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt(np.sum(diff**2, axis=2))

# Meshgrid for 2D functions
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)
```

---

## Useful Constants

```python
np.pi                      # π (3.14159...)
np.e                       # e (2.71828...)
np.inf                     # Infinity
np.nan                     # Not a Number
```

---

## Resources

- **Official Documentation**: https://numpy.org/doc/
- **User Guide**: https://numpy.org/doc/stable/user/
- **API Reference**: https://numpy.org/doc/stable/reference/
- **Tutorials**: https://numpy.org/learn/

---

## Quick Tips

1. **Always vectorize** - Avoid Python loops, use NumPy operations
2. **Use views when possible** - Faster and more memory efficient
3. **Pre-allocate arrays** - Don't grow arrays in loops
4. **Choose appropriate dtype** - Save memory with smaller types
5. **Use axis parameter** - For operations on specific dimensions
6. **Broadcasting is powerful** - Learn the rules to avoid explicit loops
7. **Check array properties** - Use `.shape`, `.dtype` to debug
8. **Copy when needed** - Use `.copy()` to avoid unintended modifications

---

**Created for comprehensive NumPy learning - From basics to advanced topics!**
