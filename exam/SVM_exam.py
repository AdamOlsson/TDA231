import numpy as np 

def kernel(xn, xm):
    return np.exp(-np.transpose(xn-xm)@(xn-xm))

x1 = np.array([0.37431335, 1.16780523])
x2 = np.array([0.73829051, -0.56552444])
x3 = np.array([-0.37910848, 0.79427185])

x4 = np.array([0.27019947, 0.51053191])
x5 = np.array([0.39835632, 0.1232738])
x6 = np.array([-0.12221539, 0.22353243])

data = np.array([x1,x2,x3,x4,x5,x6])

print('kernel(x{}, x{}): {}'.format(1,2, kernel(x1, x2)))
print('kernel(x{}, x{}): {}'.format(2,3, kernel(x2, x3)))
print('kernel(x{}, x{}): {}'.format(4,5, kernel(x4, x5)))
print('kernel(x{}, x{}): {}'.format(5,6, kernel(x5, x6)))
print('kernel(x{}, x{}): {}'.format(1,6, kernel(x1, x6)))
print('kernel(x{}, x{}): {}'.format(2,5, kernel(x2, x5)))

#for x in range(6):
#    print('x{}:\n'.format(x+1))
#    for y in range(6):
#        print('kernel(x{}, x{}): {}'.format(x+1,y+1, kernel(data[x], data[y])))
#    print('\n')
