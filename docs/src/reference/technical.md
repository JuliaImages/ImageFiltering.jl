# Technical background

## Introduction

An image filter can be represented by a function

```math
 w: \{s\in \mathbb{Z} \mid -k_1 \le s \le k_1  \} \times  \{t \in \mathbb{Z} \mid -k_2 \le t \le k_2  \}   \rightarrow \mathbb{R},
```

where ``k_i  \in \mathbb{N}`` (i = 1,2). It is common to define ``k_1 = 2a+1``
and ``k_2 = 2b + 1``, where ``a`` and ``b`` are integers, which ensures that the
filter dimensions are of odd size. Typically, ``k_1`` equals ``k_2`` and so,
dropping the subscripts, one speaks of a ``k \times k`` filter. Since the
domain of the filter represents a grid of spatial coordinates, the filter is
often called a mask and is visualized as a grid. 

For example, a ``3 \times 3`` mask can be portrayed as follows:

```math
\scriptsize
\begin{matrix}
\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(-1,-1) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}

&

\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(-1,0) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}
 &
\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(-1,1) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}
\\
\\
\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(0,-1) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}

&

\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(0,0) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}
 &
\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(0,1) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}
\\
\\
\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(1,-1) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}

&

\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(1,0) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}
 &
\boxed{
\begin{matrix}
\phantom{w(-9,-9)} \\
w(1,1) \\
\phantom{w(-9,-9)} \\
\end{matrix}
}
\end{matrix}.
```

The values of ``w(s,t)`` are referred to as *filter coefficients*.

## Convolution versus correlation

There are two fundamental and closely related operations that one regularly
performs on an image with a filter. The operations are called discrete
*correlation* and *convolution*.

The correlation operation, denoted by the symbol ``\star``,  is given in two
dimensions by the expression

```math
\begin{aligned}
g(x,y) = w(x,y) \star f(x,y) = \sum_{s = -a}^{a} \sum_{t=-b}^{b} w(s,t) f(x+s, y+t),
\end{aligned}
```
whereas the comparable convolution operation, denoted by the symbol ``\ast``,
is given in two dimensions by

```math
\begin{aligned}
h(x,y) = w(x,y) \ast f(x,y) = \sum_{s = -a}^{a} \sum_{t=-b}^{b} w(s,t) f(x-s, y-t).
\end{aligned}
```
Since a digital image is of finite extent, both of these operations are
undefined at the borders of the image. In particular, for an image of size ``M
\times N``, the function ``f(x \pm s, y \pm t)`` is only defined for ``1 \le
x \pm s \le N`` and ``1 \le y \pm t \le M``. In practice one addresses this
problem by artificially expanding the domain of the image. For example, one can
pad the image with zeros. Other padding strategies are possible, and they are
discussed in more detail in the *Options* section of this documentation.

## One-dimensional illustration

The difference between correlation and convolution is best understood with
recourse to a one-dimensional example  adapted from [1]. Suppose that a filter
``w:\{-1,0,1\}\rightarrow \mathbb{R}`` has coefficients

```math
\begin{matrix}
\boxed{1} & \boxed{2} & \boxed{3}
\end{matrix}.
```
Consider a discrete unit impulse function ``f: \{x \in \mathbb{Z} \mid 1 \le x
\le 7  \} \rightarrow \{0,1\}``  that has been padded with zeros. The function
can be visualised as an image

```math
\boxed{
\begin{matrix}
0 & \boxed{0} & \boxed{0} & \boxed{0} & \boxed{1} & \boxed{0} & \boxed{0} & \boxed{0} & 0
\end{matrix}}.
```
The correlation operation can be interpreted as sliding ``w`` along the image
and computing the sum of products at each location. For example,
```math
\begin{matrix}
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
1 & 2 & 3  &  & & & & & \\
& 1 & 2 & 3  &  & & & &  \\
& & 1 & 2 & 3  &  & & &  \\
& & & 1 & 2 & 3  &  & &  \\
& & & & 1 & 2 & 3  &  &  \\
& & & & & 1 & 2 & 3  &  \\
& & & & & & 1 & 2 & 3,
\end{matrix}
```
yields the output ``g: \{x \in \mathbb{Z} \mid 1 \le x \le 7  \} \rightarrow
\mathbb{R}``, which when visualized as a digital image, is equal to

```math
\boxed{
\begin{matrix}
\boxed{0} & \boxed{0} & \boxed{3} & \boxed{2} & \boxed{1} & \boxed{0} & \boxed{0}
\end{matrix}}.
```

The interpretation of the convolution operation is analogous to correlation,
except that the filter ``w`` has been rotated by 180 degrees. In particular,

```math
\begin{matrix}
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
3 & 2 & 1  &  & & & & & \\
& 3 & 2 & 1  &  & & & &  \\
& & 3 & 2 & 1  &  & & &  \\
& & & 3 & 2 & 1  &  & &  \\
& & & & 3 & 2 & 1  &  &  \\
& & & & & 3 & 2 & 1  &  \\
& & & & & & 3 & 2 & 1,
\end{matrix}
```

yields the output ``h: \{x \in \mathbb{Z} \mid 1 \le x \le 7  \} \rightarrow \mathbb{R}`` equal to

```math
\boxed{
\begin{matrix}
\boxed{0} & \boxed{0} & \boxed{1} & \boxed{2} & \boxed{3} & \boxed{0} & \boxed{0}
\end{matrix}}.
```

Instead of rotating the filter mask, one could instead rotate ``f`` and still
obtained the same convolution result. In fact, the conventional notation for
convolution indicates that ``f`` is flipped and not ``w``. If ``w`` is
symmetric, then convolution and correlation give the same outcome.

 ## Two-dimensional illustration

 For a two-dimensional example, suppose the filter ``w:\{-1, 0 ,1\} \times
 \{-1,0,1\} \rightarrow \mathbb{R}``  has coefficients

```math
 \begin{matrix}
 \boxed{1} & \boxed{2} & \boxed{3} \\ \\
 \boxed{4} & \boxed{5} & \boxed{6} \\ \\
 \boxed{7} & \boxed{8} & \boxed{9}
 \end{matrix},
```
 and consider a two-dimensional discrete unit impulse function

```math
 f:\{x \in \mathbb{Z} \mid 1 \le x \le 7  \} \times  \{y \in \mathbb{Z} \mid 1 \le y \le 7  \}\rightarrow \{ 0,1\}
```

 that has been padded with zeros:

```math
 \boxed{
 \begin{matrix}
   0 &        0  &        0  &        0   &        0  &        0  &   0  \\ \\
   0 & \boxed{0} & \boxed{0} & \boxed{0}  & \boxed{0} & \boxed{0} &   0  \\ \\
   0 & \boxed{0} & \boxed{0} & \boxed{0}  & \boxed{0} & \boxed{0} &   0 \\ \\
   0 & \boxed{0} & \boxed{0} & \boxed{1}  & \boxed{0} & \boxed{0} &   0 \\ \\
   0 & \boxed{0} & \boxed{0} & \boxed{0}  & \boxed{0} & \boxed{0} &   0 \\ \\
   0 & \boxed{0} & \boxed{0} & \boxed{0}  & \boxed{0} & \boxed{0} &   0 \\ \\
   0 &        0  &        0  &        0   &        0  &        0  &   0
 \end{matrix}}.
```
 The correlation operation ``w(x,y) \star f(x,y)``  yields the output

```math
 \boxed{
 \begin{matrix}
 \boxed{0} & \boxed{0}  & \boxed{0} & \boxed{0} & \boxed{0} \\ \\
 \boxed{0} &  \boxed{9} & \boxed{8} & \boxed{7} & \boxed{0} \\ \\
 \boxed{0} &  \boxed{6} & \boxed{5} & \boxed{4} & \boxed{0} \\ \\
 \boxed{0} &  \boxed{3} & \boxed{2} & \boxed{1} & \boxed{0} \\ \\
 \boxed{0} & \boxed{0}  & \boxed{0} & \boxed{0} & \boxed{0}
 \end{matrix}},
```
 whereas the convolution operation ``w(x,y) \ast f(x,y)`` produces

```math
 \boxed{
 \begin{matrix}
 \boxed{0} & \boxed{0} & \boxed{0} & \boxed{0} & \boxed{0} \\ \\
 \boxed{0} & \boxed{1} & \boxed{2} & \boxed{3} & \boxed{0}\\ \\
 \boxed{0} & \boxed{4} & \boxed{5} & \boxed{6} & \boxed{0} \\ \\
 \boxed{0} & \boxed{7} & \boxed{8} & \boxed{9} & \boxed{0} \\ \\
 \boxed{0} & \boxed{0} & \boxed{0} & \boxed{0} & \boxed{0}
 \end{matrix}}.
```

## Convolution and correlation as matrix multiplication

Discrete convolution and correlation operations can also be formulated as a
matrix multiplication, where one of the inputs is converted to a [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix)
matrix, and the other is represented as a column vector. For example, consider a
function ``f:\{x \in \mathbb{N} \mid 1 \le x \le M \} \rightarrow \mathbb{R}``
and a filter ``w: \{s \in \mathbb{N} \mid  -k_1 \le s \le k_1  \} \rightarrow
\mathbb{R}``. Then the matrix multiplication

```math
\begin{bmatrix}
w(-k_1) 	&  0	    & \ldots	& 0		   & 0			\\
\vdots 	& w(-k_1) 	& \ldots	& \vdots  & 0	        \\
w(k_1) 	    & \vdots   & \ldots	& 0		   & \vdots    \\
0 	    	& w(k_1)	& \ldots   & w(-k_1)  & 0		    \\
0 	        & 0		    & \ldots	& \vdots  & w(-k_1)	\\
\vdots     & \vdots	& \ldots	& w(k_1)   & \vdots	\\
0           & 0         & 0			& 0		   & w(k_1)
\end{bmatrix}
\begin{bmatrix}
f(1) \\
f(2) \\
f(3) \\
\vdots \\
f(M)
\end{bmatrix}
```

is equivalent to the convolution ``w(s) \ast f(x)`` assuming that the border of
``f(x)`` has been padded with zeros.

To represent multidimensional convolution as matrix multiplication one
reshapes the multidimensional arrays into column vectors and proceeds in an
analogous manner. Naturally, the result of the matrix multiplication will need
to be reshaped into an appropriate multidimensional array.
