import numpy as np
w = np.array([[1,3,3,4]])
print(w)
print(w.shape)

x = np.array([[1,2,3,4],[4,5,6,5]])
print(x)
print(x.shape)

y = np.array([[2],[1]])
print(y)
print(y.shape)


ans = np.dot(x,w.T) - y

np.dot(ans.T,x)


print(np.dot(x,w.T).T.shape)



