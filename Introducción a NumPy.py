#!/usr/bin/env python
# coding: utf-8

# # Introducción a NumPy

# [Numpy](https://numpy.org) es una librería fundamental para la computación científica con Python.
# * Proporciona arrays N-dimensionales
# * Implementa funciones matemáticas sofisticadas
# * Proporciona herramientas para integrar C/C++ y Fortran
# * Proporciona mecanismos para facilitar la realización de tareas relacionadas con álgebra lineal o números aleatorios

# ## Imports

# In[1]:


import numpy as np


# ## Arrays

# Un **array** es una estructura de datos que consiste en una colección de elementos (valores o variables), cada uno identificado por al menos un índice o clave. Un array se almacena de modo que la posición de cada elemento se pueda calcular a partir de su tupla de índice mediante una fórmula matemática. El tipo más simple de array es un array lineal, también llamado array unidimensional.

# En numpy:
# * Cada dimensión se denomina **axis**
# * El número de dimensiones se denomina **rank**
# * La lista de dimensiones con su correspondiente longitud se denomina **shape**
# * El número total de elementos (multiplicación de la longitud de las dimensiones) se denomina **size**

# In[2]:


# Array cuyos valores son todos 0
a = np.zeros((2, 4))


# In[3]:


a


# _**a**_ es un array:
# * Con dos **axis**, el primero de longitud 2 y el segundo de longitud 4
# * Con un **rank** igual a 2
# * Con un **shape** igual (2, 4)
# * Con un **size** igual a 8

# In[ ]:


a.shape


# In[ ]:


a.ndim


# In[ ]:


a.size


# ## Creación de Arrays

# In[ ]:


# Array cuyos valores son todos 0
np.zeros((2, 3, 4))


# In[ ]:


# Array cuyos valores son todos 1
np.ones((2, 3, 4))


# In[ ]:


# Array cuyos valores son todos el valor indicado como segundo parámetro de la función
np.full((2, 3, 4), 8)


# In[ ]:


# El resultado de np.empty no es predecible 
# Inicializa los valores del array con lo que haya en memoria en ese momento
np.empty((2, 3, 9))


# In[ ]:


# Inicializacion del array utilizando un array de Python
b = np.array([[1, 2, 3], [4, 5, 6]])
b


# In[ ]:


b.shape


# In[ ]:


# Creación del array utilizando una función basada en rangos
# (minimo, maximo, número elementos del array)
print(np.linspace(0, 6, 10))


# In[ ]:


# Inicialización del array con valores aleatorios
np.random.rand(2, 3, 4)


# In[ ]:


# Inicialización del array con valores aleatorios conforme a una distribución normal
np.random.randn(2, 4)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

c = np.random.randn(1000000)

plt.hist(c, bins=200)
plt.show()


# In[ ]:


# Inicialización del Array utilizando una función personalizada

def func(x, y):
    return x + 2 * y

np.fromfunction(func, (3, 5))


# ## Acceso a los elementos de un array

# ### Array unidimensional

# In[ ]:


# Creación de un Array unidimensional
array_uni = np.array([1, 3, 5, 7, 9, 11])
print("Shape:", array_uni.shape)
print("Array_uni:", array_uni)


# In[ ]:


# Accediendo al quinto elemento del Array
array_uni[4]


# In[ ]:


# Accediendo al tercer y cuarto elemento del Array
array_uni[2:4]


# In[ ]:


# Accediendo a los elementos 0, 3 y 5 del Array
array_uni[0::3]


# ### Array multidimensional

# In[ ]:


# Creación de un Array multidimensional
array_multi = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("Shape:", array_multi.shape)
print("Array_multi:\n", array_multi)


# In[ ]:


# Accediendo al cuarto elemento del Array
array_multi[0, 3]


# In[ ]:


# Accediendo a la segunda fila del Array
array_multi[1, :]


# In[ ]:


# Accediendo al tercer elemento de las dos primeras filas del Array
array_multi[0:2, 2]


# ## Modificación de un Array

# In[ ]:


# Creación de un Array unidimensional inicializado con el rango de elementos 0-27
array1 = np.arange(28)
print("Shape:", array1.shape)
print("Array 1:", array1)


# In[ ]:


# Cambiar las dimensiones del Array y sus longitudes
array1.shape = (7, 4)
print("Shape:", array1.shape)
print("Array 1:\n", array1)


# In[ ]:


# El ejemplo anterior devuelve un nuevo Array que apunta a los mismos datos. 
# Importante: Modificaciones en un Array, modificaran el otro Array
array2 = array1.reshape(4, 7)
print("Shape:", array2.shape)
print("Array 2:\n", array2)


# In[ ]:


# Modificación del nuevo Array devuelto
array2[0, 3] = 20
print("Array 2:\n", array2)


# In[ ]:


print("Array 1:\n", array1)


# In[ ]:


# Desenvuelve el Array, devolviendo un nuevo Array de una sola dimension
# Importante: El nuevo array apunta a los mismos datos
print("Array 1:", array1.ravel())


# ## Operaciones aritméticas con Arrays

# In[ ]:


# Creación de dos Arrays unidimensionales
array1 = np.arange(2, 18, 2)
array2 = np.arange(8)
print("Array 1:", array1)
print("Array 2:", array2)


# In[ ]:


# Suma
print(array1 + array2)


# In[ ]:


# Resta
print(array1 - array2)


# In[ ]:


# Multiplicacion
# Importante: No es una multiplicación de matrices
print(array1 * array2)


# ## Broadcasting

# Si se aplican operaciones aritméticas sobre Arrays que no tienen la misma forma (shape) Numpy aplica un propiedad que se denomina Broadcasting.

# In[ ]:


# Creación de dos Arrays unidimensionales
array1 = np.arange(5)
array2 = np.array([3])
print("Shape Array 1:", array1.shape)
print("Array 1:", array1)
print()
print("Shape Array 2:", array2.shape)
print("Array 2:", array2)


# In[ ]:


# Suma de ambos Arrays
array1 + array2


# In[ ]:


# Creación de dos Arrays multidimensional y unidimensional
array1 = np.arange(6)
array1.shape = (2, 3)
array2 = np.arange(6, 18, 4)
print("Shape Array 1:", array1.shape)
print("Array 1:\n", array1)
print()
print("Shape Array 2:", array2.shape)
print("Array 2:", array2)


# In[ ]:


# Suma de ambos Arrays
array1 + array2


# ## Funciones estadísticas sobre Arrays

# In[ ]:


# Creación de un Array unidimensional
array1 = np.arange(1, 20, 2)
print("Array 1:", array1)


# In[ ]:


# Media de los elementos del Array
array1.mean()


# In[ ]:


# Suma de los elementos del Array
array1.sum()


# Funciones universales eficientes proporcionadas por numpy: **ufunc**

# In[ ]:


# Cuadrado de los elementos del Array
np.square(array1)


# In[ ]:


# Raiz cuadrada de los elementos del Array
np.sqrt(array1)


# In[ ]:


# Exponencial de los elementos del Array
np.exp(array1)


# In[ ]:


# log de los elementos del Array
np.log(array1)


# In[ ]:




