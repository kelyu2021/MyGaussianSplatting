Inside the rasterizer, depth is computed as an opacity weighted sum of per Gaussian depths along each ray, essentially the same alpha-compositing used for color, but accumulating the view space z-coordinate of each Gaussian instead of its color:
$$
\text{depth}(p) =
\frac{
\sum_i \alpha_i \left( \prod_{j<i} (1 - \alpha_j) \right) z_i
}{
\sum_i \alpha_i \left( \prod_{j<i} (1 - \alpha_j) \right) + \epsilon
}
$$
where $$z_i$$ is the depth of Gaussian i in camera space and $$α_i$$ is its blended opacity contribution at pixel p. The denominator is the accumulated alpha (acc), which is why acc is passed alongside depth into depth_mono_loss — to normalize or weight the supervision.