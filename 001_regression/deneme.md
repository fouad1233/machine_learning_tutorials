
Tabii! İşte, denklemlerin LaTeX formatında uygun bir şekilde yazılmış hali. Bu formatı Word veya LaTeX destekleyen diğer dokümanlarda kullanabilirsiniz.

### 1. Genel Polinom Regresyon Modeli:

\[
y_i = \theta_0 + \theta_1 x_i + \theta_2 x_i^2 + \dots + \theta_m x_i^m + \varepsilon_i
\]

### 2. Matris Formu:

\[
\mathbf{y} = g(\mathbf{X}|\theta) + \boldsymbol{\varepsilon}
\]

Burada:

\[
g(\mathbf{X}|\theta) =
\begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^m \\
1 & x_2 & x_2^2 & \dots & x_2^m \\
1 & x_3 & x_3^2 & \dots & x_3^m \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \dots & x_n^m
\end{bmatrix}
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_m
\end{bmatrix}
\]

### 3. En Küçük Kareler Yöntemi:

Hata terimi:

\[
\boldsymbol{\varepsilon} = \mathbf{y} - g(\mathbf{X}|\theta)
\]

Hata fonksiyonu:

\[
S(\theta) = (\mathbf{y} - g(\mathbf{X}|\theta))^\top (\mathbf{y} - g(\mathbf{X}|\theta))
\]

Türev işlemi ve normal denklem:

\[
\mathbf{X}^\top \mathbf{X} \theta = \mathbf{X}^\top \mathbf{y}
\]

Son çözüm:

\[
\hat{\theta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
\]

Bu formatta, denklemleri kolayca Word veya LaTeX belgelerine kopyalayabilirsiniz.
