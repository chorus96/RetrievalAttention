![Logo](https://picgo-1258602555.cos.ap-nanjing.myqcloud.com/icon.png)

# [latex2sympy2](https://github.com/OrangeX4/latex2sympy)

## 소개

`latex2sympy2`는 **LaTeX 수학 표현식**을 파싱하여 동등한 **SymPy 형식**으로 변환합니다. latex2sympy2는 [augustt198/latex2sympy](https://github.com/augustt198/latex2sympy)와 [purdue-tlt / latex2sympy](https://github.com/purdue-tlt/latex2sympy)를 기반으로 개작되었습니다.

이 프로젝트는 [Latex Sympy Calculator](https://marketplace.visualstudio.com/items?itemName=OrangeX4.latex-sympy-calculator)라는 VS Code 확장의 일부입니다. latex이나 markdown으로 작성하는 사람들이 수학 표현식을 쓰면서 무언가를 계산할 수 있는 기능을 제공하도록 설계되었습니다.

파서를 생성하는 데 [ANTLR](http://www.antlr.org/)이 사용됩니다.

## 기능

* **산술:** 덧셈 (+), 뺄셈 (-), 점곱 (·), 벡터곱 (×), 분수 (/), 거듭제곱 (^), 절댓값 (|x|), 제곱근 (√) 등...
* **문자:** a - z, A - Z, α - ω, 아래첨자 (x_1), 액센트 바 (ā) 등...
* **일반 함수:** gcd, lcm, floor, ceil, max, min, log, ln, exp, sin, cos, tan, csc, sec, cot, arcsin, sinh, arsinh 등...
* **함수 기호:** f(x), f(x-1,), g(x,y) 등...
* **미적분:** 극한 ($lim_{n\to\infty}$), 미분 ($\frac{d}{dx}(x^2+x)$), 적분 ($\int xdx$) 등...
* **선형대수:** 행렬(Matrix), 행렬식(Determinant), 전치(Transpose), 역행렬(Inverse), 기본 변형(Elementary Transformation) 등...
* **기타:** 이항 계수(Binomial)...

**주의:** 행렬식, 전치 행렬, 기본 변형을 변환할 때 일부 되돌릴 수 없는 계산을 수행합니다...

## 설치

```
pip install latex2sympy2
```

**요구 사항:** `sympy`와 `antlr4-python3-runtime` 패키지.

## 사용법

### 기본

Python에서:

```python
from latex2sympy2 import latex2sympy, latex2latex

tex = r"\frac{d}{dx}(x^{2}+x)"
# 또는 'd' 대신 '\mathrm{d}'를 사용할 수 있습니다
latex2sympy(tex)
# => "Derivative(x**2 + x, x)"
latex2latex(tex)
# => "2 x + 1"
```

### 예시

|LaTeX|변환된 SymPy|계산된 LaTeX|
|-----|-----|---------------|
|`x^{3}` $x^{3}$| `x**3`|`x^{3}` $x^{3}$|
|`\frac{d}{dx} tx` $\frac{d}{dx}tx$|`Derivative(x*t, x)`|`t` $t$|
|`\sum_{i = 1}^{n} i` $\sum_{i = 1}^{n} i$|`Sum(i, (i, 1, n))`|`\frac{n \left(n + 1\right)}{2}` $\frac{n \left(n + 1\right)}{2}$|
|`\int_{a}^{b} \frac{dt}{t}`|`Integral(1/t, (t, a, b))`|`-\log{(a)} + \log{(b)}` $-\log{(a)} + \log{(b)}$|
|`(2x^3 - x + z)|_{x=3}` $(2x^3 - x + z)\|_{x=3}$|`z + 51`| `z + 51` $z + 51$ |

수학 공식을 읽고 싶다면 [GitNotes](https://notes.orangex4.cool/?git=github&github=OrangeX4/latex2sympy)를 클릭하면 됩니다.

### 방정식 풀이

``` latex
# 변환 전
x + y = 1

# 변환 후
[ y = 1 - x, \  x = 1 - y]
```

### 특정 값 대입(Eval At)

``` latex
# 변환 전
(x+2)|_{x=y+1}

# 변환 후
y + 3
```

### 행렬(Matrix)

#### 단위 행렬

```
tex = r"\bm{I}_3"
latex2sympy(tex)
# => "Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])"
```

#### 행렬식

``` python
from latex2sympy2 import latex2sympy

tex = r"\begin{vmatrix} x & 0 & 0 \\ 0 & x & 0 \\ 0 & 0 & x \end{vmatrix}"
latex2sympy(tex)
# => "x^{3}"
```

#### 전치

``` python
from latex2sympy2 import latex2sympy

tex = r"\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}^T"
# 또는 "\begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix}'"를 사용할 수 있습니다
latex2sympy(tex)
# => "Matrix([[1, 4, 7], [2, 5, 8], [3, 6, 9]])"
```

#### 기본 변형

``` python
from latex2sympy2 import latex2sympy

matrix = r'''
    \begin{pmatrix}
        1 & 2 & 3 \\ 
        4 & 5 & 6 \\
        7 & 8 & 9 \\ 
    \end{pmatrix}
'''

# "\xrightarrow{kr_n}" 문법으로 행을 스칼라배합니다
tex = matrix + r'\xrightarrow{3r_1}'
latex2sympy(tex)
# => "Matrix([[3, 6, 9], [4, 5, 6], [7, 8, 9]])"

# "\xrightarrow{c_1<=>c_2}" 문법으로 열을 서로 바꿉니다
# 물론 "<=>" 대신 "\leftrightarrow"를 사용할 수 있습니다
tex = matrix + r'\xrightarrow{c_1<=>c_2}'
latex2sympy(tex)
# => "Matrix([[2, 1, 3], [5, 4, 6], [8, 7, 9]])"

# 두 번째 행을 스칼라배하여 첫 번째 행에 더합니다
# "\xrightarrow{r_1+kr_2}" 문법 사용
tex = matrix + r'\xrightarrow{r_1+kr_2}'
latex2sympy(tex)
# => "Matrix([[4*k + 1, 5*k + 2, 6*k + 3], [4, 5, 6], [7, 8, 9]])"

# 콤마 ","와 "\xrightarrow[4r_3]{2r_1, 3r_2}" 문법으로
# 변형을 조합할 수 있습니다
# "{}"의 우선순위가 "[]"보다 높다는 점을 기억하세요
tex = matrix + r'\xrightarrow[4r_3]{2r_1, 3r_2}'
latex2sympy(tex)
# => "Matrix([[2, 4, 6], [12, 15, 18], [28, 32, 36]])"
```

### 변수(Variances)

``` python
from latex2sympy2 import latex2sympy, variances, var, set_variances

# x에 값 1을 할당합니다
latex2sympy(r"x = 1")

# x에 차원이 n x m인 행렬 심볼을 할당합니다
latex2sympy(r"x \in \mathbb{R}^{n \times m}")

# x + y를 계산합니다
latex2sympy(r"x + y")
# => "y + 1"

# 모든 변수를 가져옵니다
print(variances)
# => "{x: 1}"

# "x"의 값을 가져옵니다
print(var["x"])
# => "1"

# 모든 변수를 초기화합니다
set_variances({})
latex2sympy(r"x + y")
# => "x + y"
```

### 복소수 지원

``` python
from latex2sympy2 import set_real

set_real(False)
```


## 기여

새로운 문법을 추가하고 싶다면 [OrangeX4/latex2sympy](https://github.com/OrangeX4/latex2sympy)에서 코드를 포크할 수 있습니다.

* 파서 문법을 수정하려면 `PS.g4`의 기존 구조를 살펴보세요.
* 각 문법에 연결된 동작을 수정하려면 `latex2sympy.py`를 확인하세요.

기여자를 환영합니다! 풀 리퀘스트나 이슈를 자유롭게 열어 주세요.
