# Padding arrays

## Introduction

The `padarray()` function generates a padded image from an array `img` and a specification
`border` of the boundary conditions and amount of padding to add.

```julia
padarray([T], img, border)
```

The function returns a new image that is an expansion of the input image, in
which additional pixels are derived from the border of the input image using the
extrapolation scheme specified by `border`.

The function supports one, two or multi-dimensional images. You can specify the
element type `T` of the output image.

### The `Pad` type

The type `Pad` designates the form of padding which should be used to
extrapolate pixels beyond the boundary of an image. Instances must set `style`,
a Symbol specifying the boundary conditions of the image.

The symbol must be one of:

- `:replicate` (repeat edge values to infinity),
- `:circular` (image edges "wrap around"),
- `:symmetric` (the image reflects relative to a position between pixels),
- `:reflect` (the image reflects relative to the edge itself).

Refer to the documentation of [`Pad`](@ref) for more details and examples for
each option.

### The `Fill` type

The type `Fill` designates a particular value which will be used to
extrapolate pixels beyond the boundary of an image. 

Refer to the documentation of [`Fill`](@ref) for more details and illustrations.

## 2D Examples

The following examples show the effects of modifying the input array:

```math
\mathbf{A} =
\boxed{
\begin{matrix}
 1  & 2  &  3  &  4 & 5  & 6 \\
 2  & 4  &  6  &  8 & 10 & 12 \\
 3  & 6  &  9  & 12 & 15 & 18 \\
 4  & 8  & 12  & 16 & 20 & 24 \\
 5  & 10 & 15  & 20 & 25 & 30 \\
 6  & 12 & 18  & 24 & 30 & 36
 \end{matrix}}.
```

### Examples with `Pad`

The command `padarray(A, Pad(:replicate, 4, 4))` yields

```math
\boxed{
\begin{array}{ccccccccccccc}
1 & 1 & 1 & 1 &         1   &          2   &          3   &          4   &          5   &          6   &  6  &  6  &  6  &  6 \\
1 & 1 & 1 & 1 &         1   &          2   &          3   &          4   &          5   &          6   &  6  &  6  &  6  &  6 \\
1 & 1 & 1 & 1 &         1   &          2   &          3   &          4   &          5   &          6   &  6  &  6  &  6  &  6 \\
1 & 1 & 1 & 1 &         1   &          2   &          3   &          4   &          5   &          6   &  6  &  6  &  6  &  6 \\
1 & 1 & 1 & 1 &  \boxed{1}  &   \boxed{2}  &   \boxed{3}  &   \boxed{4}  &   \boxed{5}  &   \boxed{6}  &  6  &  6  &  6  &  6 \\
2 & 2 & 2 & 2 &  \boxed{2}  &   \boxed{4}  &   \boxed{6}  &   \boxed{8}  &  \boxed{10}  &  \boxed{12}  & 12  & 12  & 12  & 12 \\
3 & 3 & 3 & 3 &  \boxed{3}  &   \boxed{6}  &   \boxed{9}  &  \boxed{12}  &  \boxed{15}  &  \boxed{18}  & 18  & 18  & 18  & 18 \\
4 & 4 & 4 & 4 &  \boxed{4}  &   \boxed{8}  &  \boxed{12}  &  \boxed{16}  &  \boxed{20}  &  \boxed{24}  & 24  & 24  & 24  & 24 \\
5 & 5 & 5 & 5 &  \boxed{5}  &  \boxed{10}  &  \boxed{15}  &  \boxed{20}  &  \boxed{25}  &  \boxed{30}  & 30  & 30  & 30  & 30 \\
6 & 6 & 6 & 6 &  \boxed{6}  &  \boxed{12}  &  \boxed{18}  &  \boxed{24}  &  \boxed{30}  &  \boxed{36}  & 36  & 36  & 36  & 36 \\
6 & 6 & 6 & 6 &         6   &         12   &         18   &         24   &         30   &         36   & 36  & 36  & 36  & 36 \\
6 & 6 & 6 & 6 &         6   &         12   &         18   &         24   &         30   &         36   & 36  & 36  & 36  & 36 \\
6 & 6 & 6 & 6 &         6   &         12   &         18   &         24   &         30   &         36   & 36  & 36  & 36  & 36 \\
6 & 6 & 6 & 6 &         6   &         12   &         18   &         24   &         30   &         36   & 36  & 36  & 36  & 36
 \end{array}
}.
```

The command `padarray(A, Pad(:circular,4,4))` yields

```math
\boxed{
\begin{array}{ccccccccccccc}
9  & 12 & 15 & 18 &         3  &         6   &         9   &         12  &          15  &         18  & 3 &  6 &  9 & 12 \\
12 & 16 & 20 & 24 &         4  &         8   &        12   &         16  &          20  &         24  & 4 &  8 & 12 & 16 \\
15 & 20 & 25 & 30 &         5  &        10   &        15   &         20  &          25  &         30  & 5 & 10 & 15 & 20 \\
18 & 24 & 30 & 36 &         6  &        12   &        18   &         24  &          30  &         36  & 6 & 12 & 18 & 24 \\
3  &  4 &  5 &  6 &  \boxed{1} &  \boxed{2}  &  \boxed{3}  &  \boxed{4}  &  \boxed{5}   &  \boxed{6}  & 1 &  2 &  3 &  4 \\
6  &  8 & 10 & 12 &  \boxed{2} &  \boxed{4}  &  \boxed{6}  &  \boxed{8}  &  \boxed{10}  &  \boxed{12} & 2 &  4 &  6 &  8 \\
9  & 12 & 15 & 18 &  \boxed{3} &  \boxed{6}  &  \boxed{9}  &  \boxed{12} &  \boxed{15}  &  \boxed{18} & 3 &  6 &  9 & 12 \\
12 & 16 & 20 & 24 &  \boxed{4} &  \boxed{8}  &  \boxed{12} &  \boxed{16} &  \boxed{20}  &  \boxed{24} & 4 &  8 & 12 & 16 \\
15 & 20 & 25 & 30 &  \boxed{5} &  \boxed{10} &  \boxed{15} &  \boxed{20} &  \boxed{25}  &  \boxed{30} & 5 & 10 & 15 & 20 \\
18 & 24 & 30 & 36 &  \boxed{6} &  \boxed{12} &  \boxed{18} &  \boxed{24} &  \boxed{30}  &  \boxed{36} & 6 & 12 & 18 & 24 \\
3  &  4 &  5 &  6 &         1  &          2  &          3  &          4  &           5  &          6  & 1 &  2 &  3 &  4 \\
6  &  8 & 10 & 12 &         2  &          4  &          6  &          8  &          10  &         12  & 2 &  4 &  6 &  8 \\
9  & 12 & 15 & 18 &         3  &          6  &          9  &         12  &          15  &         18  & 3 &  6 &  9 & 12 \\
12 & 16 & 20 & 24 &         4  &          8  &         12  &         16  &          20  &         24  & 4 &  8 & 12 & 16
\end{array}
}.
```

The command `padarray(A, Pad(:symmetric,4,4))` yields

```math
\boxed{
\begin{array}{ccccccccccccc}
16 & 12 &  8 & 4 &         4  &          8  &         12  &          16 &          20 &         24  & 24 & 20 & 16 & 12 \\
12 &  9 &  6 & 3 &         3  &          6  &         9   &          12 &          15 &         18  & 18 & 15 & 12 &  9 \\
 8 &  6 &  4 & 2 &         2  &          4  &         6   &          8  &          10 &         12  & 12 & 10 &  8 &  6 \\
 4 &  3 &  2 & 1 &         1  &          2  &         3   &          4  &          5  &         6   &  6 &  5 &  4 &  3 \\
 4 &  3 &  2 & 1 &  \boxed{1} &   \boxed{2} &  \boxed{3}  &   \boxed{4} &  \boxed{5}  &  \boxed{6}  &  6 &  5 &  4 &  3 \\
 8 &  6 &  4 & 2 &  \boxed{2} &   \boxed{4} &  \boxed{6}  &   \boxed{8} &  \boxed{10} &  \boxed{12} & 12 & 10 &  8 &  6 \\
12 &  9 &  6 & 3 &  \boxed{3} &   \boxed{6} &  \boxed{9}  &  \boxed{12} &  \boxed{15} &  \boxed{18} & 18 & 15 & 12 &  9 \\
16 & 12 &  8 & 4 &  \boxed{4} &   \boxed{8} &  \boxed{12} &  \boxed{16} &  \boxed{20} &  \boxed{24} & 24 & 20 & 16 & 12 \\
20 & 15 & 10 & 5 &  \boxed{5} &  \boxed{10} &  \boxed{15} &  \boxed{20} &  \boxed{25} &  \boxed{30} & 30 & 25 & 20 & 15 \\
24 & 18 & 12 & 6 &  \boxed{6} &  \boxed{12} &  \boxed{18} &  \boxed{24} &  \boxed{30} &  \boxed{36} & 36 & 30 & 24 & 18 \\
24 & 18 & 12 & 6 &         6  &         12  &         18  &         24  &         30  &         36  & 36 & 30 & 24 & 18 \\
20 & 15 & 10 & 5 &         5  &         10  &         15  &         20  &         25  &         30  & 30 & 25 & 20 & 15 \\
16 & 12 &  8 & 4 &         4  &          8  &         12  &         16  &         20  &         24  & 24 & 20 & 16 & 12 \\
12 &  9 &  6 & 3 &         3  &          6  &          9  &         12  &         15  &         18  & 18 & 15 & 12 &  9
\end{array}
}.
```

The command `padarray(A, Pad(:reflect,4,4))` yields

```math
\boxed{
\begin{array}{ccccccccccccc}
25 & 20 & 15 & 10 &         5  &         10  &         15   &         20  &          25  &         30  & 25 & 20 & 15 & 10 \\
20 & 16 & 12 &  8 &         4  &         8   &         12   &         16  &          20  &         24  & 20 & 16 & 12 &  8 \\
15 & 12 &  9 &  6 &         3  &         6   &          9   &         12  &          15  &         18  & 15 & 12 &  9 &  6 \\
10 &  8 &  6 &  4 &         2  &         4   &          6   &         8   &          10  &         12  & 10 &  8 &  6 &  4 \\
5  &  4 &  3 &  2 &  \boxed{1} &  \boxed{2}  &   \boxed{3}  &  \boxed{4}  &   \boxed{5}  &  \boxed{6}  &  5 &  4 &  3 &  2 \\
10 &  8 &  6 &  4 &  \boxed{2} &  \boxed{4}  &   \boxed{6}  &  \boxed{8}  &   \boxed{10} &  \boxed{12} & 10 &  8 &  6 &  4 \\
15 & 12 &  9 &  6 &  \boxed{3} &  \boxed{6}  &   \boxed{9}  &  \boxed{12} &   \boxed{15} &  \boxed{18} & 15 & 12 &  9 &  6 \\
20 & 16 & 12 &  8 &  \boxed{4} &  \boxed{8}  &   \boxed{12} &  \boxed{16} &   \boxed{20} &  \boxed{24} & 20 & 16 & 12 &  8 \\
25 & 20 & 15 & 10 &  \boxed{5} &  \boxed{10} &   \boxed{15} &  \boxed{20} &   \boxed{25} &  \boxed{30} & 25 & 20 & 15 & 10 \\
30 & 24 & 18 & 12 &  \boxed{6} &  \boxed{12} &   \boxed{18} &  \boxed{24} &   \boxed{30} &  \boxed{36} & 30 & 24 & 18 & 12 \\
25 & 20 & 15 & 10 &         5  &         10  &          15  &         20  &          25  &         30  & 25 & 20 & 15 & 10 \\
20 & 16 & 12 &  8 &         4  &         8   &          12  &         16  &          20  &         24  & 20 & 16 & 12 &  8 \\
15 & 12 &  9 &  6 &         3  &         6   &           9  &         12  &          15  &         18  & 15 & 12 &  9 &  6 \\
10 &  8 &  6 &  4 &         2  &         4   &           6  &          8  &          10  &         12  & 10 &  8 &  6 &  4
\end{array}
}.
```

### Examples with `Fill`

The command `padarray(A, Fill(0,(4,4),(4,4)))` yields

```math
\boxed{
\begin{array}{ccccccccccccc}
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &  \boxed{1} &  \boxed{2}  &  \boxed{3}  &  \boxed{4}  &  \boxed{5}  &   \boxed{6}  & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &  \boxed{2} &  \boxed{4}  &  \boxed{6}  &  \boxed{8}  &  \boxed{10} &   \boxed{12} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &  \boxed{3} &  \boxed{6}  &  \boxed{9}  &  \boxed{12} &  \boxed{15} &   \boxed{18} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &  \boxed{4} &  \boxed{8}  &  \boxed{12} &  \boxed{16} &  \boxed{20} &   \boxed{24} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &  \boxed{5} &  \boxed{10} &  \boxed{15} &  \boxed{20} &  \boxed{25} &   \boxed{30} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &  \boxed{6} &  \boxed{12} &  \boxed{18} &  \boxed{24} &  \boxed{30} &   \boxed{36} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 &         0  &         0   &         0   &         0   &         0   &          0   & 0 & 0 & 0 & 0
\end{array}
}.
```

## 3D Examples

Each example is based on a multi-dimensional array ``\mathsf{A} \in\mathbb{R}^{2 \times 2 \times 2}`` given by:

```math
\mathsf{A}(:,:,1) =
\boxed{
\begin{array}{cc}
1 & 2 \\
3 & 4
\end{array}}
\quad
\text{and}
\quad
\mathsf{A}(:,:,2) =
\boxed{
\begin{array}{cc}
5 & 6 \\
7 & 8
\end{array}}.
```

Note that each example will yield a new multi-dimensional array ``\mathsf{A}'
\in \mathbb{R}^{4 \times 4 \times 4}`` of type `OffsetArray`, where prepended
dimensions may be negative or start from zero.

### Examples with `Pad`

The command `padarray(A, Pad(:replicate, 1, 1, 1))` gives:

```math
\begin{aligned}
\mathsf{A}'(:,:,0) & =
\boxed{
\begin{array}{cccc}
1 & 1 & 2 & 2 \\
1 & 1 & 2 & 2 \\
3 & 3 & 4 & 4 \\
3 & 3 & 4 & 4
\end{array}}
&
\mathsf{A}'(:,:,1) & =
\boxed{
\begin{array}{cccc}
1 &         1  &         2  & 2 \\
1 &  \boxed{1} &  \boxed{2} & 2 \\
3 &  \boxed{3} &  \boxed{4} & 4 \\
3 &         3  &         4  & 4
\end{array}} \\
\mathsf{A}'(:,:,2) & =
\boxed{
\begin{array}{cccc}
5 &         5  &         6  & 6 \\
5 &  \boxed{5} &  \boxed{6} & 6 \\
7 &  \boxed{7} &  \boxed{8} & 8 \\
7 &         7  &         8  & 8
\end{array}}
&
\mathsf{A}'(:,:,3) & =
\boxed{
\begin{array}{cccc}
5 & 5 & 6 & 6 \\
5 & 5 & 6 & 6 \\
7 & 7 & 8 & 8 \\
7 & 7 & 8 & 8
\end{array}}
\end{aligned}
.
```

The command `padarray(A, Pad(:circular, 1, 1, 1))` gives:

```math
\begin{aligned}
\mathsf{A}'(:,:,0) & =
\boxed{
\begin{array}{cccc}
8 & 7 & 8 & 7 \\
6 & 5 & 6 & 5 \\
8 & 7 & 8 & 7 \\
6 & 5 & 6 & 5
\end{array}}
&
\mathsf{A}'(:,:,1) & =
\boxed{
\begin{array}{cccc}
4 &         3  &         4  & 3 \\
2 &  \boxed{1} &  \boxed{2} & 1 \\
4 &  \boxed{3} &  \boxed{4} & 3 \\
2 &         1  &         2  & 1
\end{array}} \\
\mathsf{A}'(:,:,2) & =
\boxed{
\begin{array}{cccc}
8 &         7  &         8  & 7 \\
6 &  \boxed{5} &  \boxed{6} & 5 \\
8 &  \boxed{7} &  \boxed{8} & 7 \\
6 &         5  &         6  & 5
\end{array}}
&
\mathsf{A}'(:,:,3) & =
\boxed{
\begin{array}{cccc}
4 & 3 & 4 & 3 \\
2 & 1 & 2 & 1 \\
4 & 3 & 4 & 3 \\
2 & 1 & 2 & 1
\end{array}}
\end{aligned}
.
```

The command `padarray(A,Pad(:symmetric, 1, 1, 1))` gives:

```math
\begin{aligned}
\mathsf{A}'(:,:,0) & =
\boxed{
\begin{array}{cccc}
1 & 1 & 2 & 2 \\
1 & 1 & 2 & 2 \\
3 & 3 & 4 & 4 \\
3 & 3 & 4 & 4
\end{array}}
&
\mathsf{A}'(:,:,1) & =
\boxed{
\begin{array}{cccc}
1 &         1  &         2  & 2 \\
1 &  \boxed{1} &  \boxed{2} & 2 \\
2 &  \boxed{3} &  \boxed{4} & 4 \\
2 &         3  &         4  & 4
\end{array}} \\
\mathsf{A}'(:,:,2) & =
\boxed{
\begin{array}{cccc}
5 &         5  &         6  & 6 \\
5 &  \boxed{5} &  \boxed{6} & 6 \\
7 &  \boxed{7} &  \boxed{8} & 8 \\
7 &         7  &         8  & 8
\end{array}}
&
\mathsf{A}'(:,:,3) & =
\boxed{
\begin{array}{cccc}
5 & 5 & 6 & 6 \\
5 & 5 & 6 & 6 \\
7 & 7 & 8 & 8 \\
7 & 7 & 8 & 8
\end{array}}
\end{aligned}
.
```

The command `padarray(A, Pad(:reflect, 1, 1, 1))` gives:

```math
\begin{aligned}
\mathsf{A}'(:,:,0) & =
\boxed{
\begin{array}{cccc}
8 & 7 & 8 & 7 \\
6 & 5 & 6 & 5 \\
8 & 7 & 8 & 7 \\
6 & 5 & 6 & 5
\end{array}}
&
\mathsf{A}'(:,:,1) & =
\boxed{
\begin{array}{cccc}
4 &         3  &         4  & 3 \\
2 &  \boxed{1} &  \boxed{2} & 1 \\
4 &  \boxed{3} &  \boxed{4} & 3 \\
2 &         1  &         2  & 1
\end{array}} \\
\mathsf{A}'(:,:,2) & =
\boxed{
\begin{array}{cccc}
8 &         7  &         8  & 7 \\
6 &  \boxed{5} &  \boxed{6} & 5 \\
8 &  \boxed{7} &  \boxed{8} & 7 \\
6 &         5  &         6  & 5
\end{array}}
&
\mathsf{A}'(:,:,3) & =
\boxed{
\begin{array}{cccc}
4 & 3 & 4 & 3 \\
2 & 1 & 2 & 1 \\
4 & 3 & 4 & 3 \\
2 & 1 & 2 & 1
\end{array}}
\end{aligned}
.
```

### Examples with `Fill`

The command `padarray(A, Fill(0, (1, 1, 1)))` gives:

```math
\begin{aligned}
\mathsf{A}'(:,:,0) & =
\boxed{
\begin{array}{cccc}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{array}}
&
\mathsf{A}'(:,:,1) & =
\boxed{
\begin{array}{cccc}
0 &         0  &         0  & 0 \\
0 &  \boxed{1} &  \boxed{2} & 0 \\
0 &  \boxed{3} &  \boxed{4} & 0 \\
0 &         0  &         0  & 0
\end{array}} \\
\mathsf{A}'(:,:,2) & =
\boxed{
\begin{array}{cccc}
0 &         0  &         0  & 0 \\
0 &  \boxed{5} &  \boxed{6} & 0 \\
0 &  \boxed{7} &  \boxed{8} & 0 \\
0 &         0  &         0  & 0
\end{array}}
&
\mathsf{A}'(:,:,3) & =
\boxed{
\begin{array}{cccc}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{array}}
\end{aligned}
.
```

## BorderArray

```
BorderArray(inner::AbstractArray, border::AbstractBorder) <: AbstractArray
```

Construct a thin wrapper around the array inner, with given
border. No data is copied in the constructor. Instead, border
values are computed on the fly in `getindex` calls. 

Useful for stencil computations.
