// Octopus Hybrid AI Engine -- linear algebra kernels.
// SPDX-License-Identifier: MIT
#include "octopus/numerics.hpp"

#include <cmath>
#include <algorithm>

namespace oct::numerics::linalg {

double inf_norm(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

// LU decomposition with partial pivoting, row-major n x n.
// Returns permutation in piv[] and whether the matrix is singular.
static bool lu_factor(std::vector<double>& A, int64_t n, std::vector<int64_t>& piv, int64_t* sign) {
    if (sign) *sign = 1;
    piv.resize(size_t(n));
    for (int64_t i = 0; i < n; ++i) piv[size_t(i)] = i;
    for (int64_t k = 0; k < n; ++k) {
        int64_t p = k;
        double best = std::abs(A[size_t(k) * size_t(n) + size_t(k)]);
        for (int64_t i = k + 1; i < n; ++i) {
            double v = std::abs(A[size_t(i) * size_t(n) + size_t(k)]);
            if (v > best) { best = v; p = i; }
        }
        if (best == 0.0) return false;
        if (p != k) {
            for (int64_t j = 0; j < n; ++j)
                std::swap(A[size_t(p) * size_t(n) + size_t(j)], A[size_t(k) * size_t(n) + size_t(j)]);
            std::swap(piv[size_t(p)], piv[size_t(k)]);
            if (sign) *sign = -*sign;
        }
        const double akk = A[size_t(k) * size_t(n) + size_t(k)];
        for (int64_t i = k + 1; i < n; ++i) {
            const double lik = A[size_t(i) * size_t(n) + size_t(k)] / akk;
            A[size_t(i) * size_t(n) + size_t(k)] = lik;
            if (lik != 0.0) {
                for (int64_t j = k + 1; j < n; ++j)
                    A[size_t(i) * size_t(n) + size_t(j)] -=
                        lik * A[size_t(k) * size_t(n) + size_t(j)];
            }
        }
    }
    return true;
}

static std::vector<double> lu_solve_factored(const std::vector<double>& LU, int64_t n,
                                             const std::vector<int64_t>& piv,
                                             const std::vector<double>& b) {
    const size_t ns = size_t(n);
    std::vector<double> y(ns);
    for (int64_t i = 0; i < n; ++i) y[size_t(i)] = b[size_t(piv[size_t(i)])];
    // Forward substitution (unit lower triangular).
    for (int64_t i = 1; i < n; ++i) {
        double s = y[size_t(i)];
        for (int64_t j = 0; j < i; ++j) s -= LU[size_t(i) * size_t(n) + size_t(j)] * y[size_t(j)];
        y[size_t(i)] = s;
    }
    // Back substitution.
    for (int64_t i = n - 1; i >= 0; --i) {
        double s = y[size_t(i)];
        for (int64_t j = i + 1; j < n; ++j) s -= LU[size_t(i) * size_t(n) + size_t(j)] * y[size_t(j)];
        const double u = LU[size_t(i) * size_t(n) + size_t(i)];
        if (u == 0.0) return {};
        y[size_t(i)] = s / u;
    }
    return y;
}

Outcome<std::vector<double>> solve(std::vector<double> A, std::vector<double> b, int64_t n,
                                   double* residual_out) {
    if (n <= 0 || int64_t(A.size()) != n * n || int64_t(b.size()) != n)
        return Status::invalid("linalg::solve: dimension mismatch");
    auto LU = A;
    std::vector<int64_t> piv;
    int64_t sign = 1;
    if (!lu_factor(LU, n, piv, &sign))
        return Status::rejected("linalg::solve: matrix is singular (exact zero pivot)");
    auto x = lu_solve_factored(LU, n, piv, b);
    if (int64_t(x.size()) != n) return Status::internal("linalg::solve: back-substitution failed");

    // Residual is the honest quality measure: it is cheap and scale-aware.
    double rmax = 0.0, amax = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0, c = 0.0, amax_row = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double t = s + A[size_t(i) * size_t(n) + size_t(j)] * x[size_t(j)];
            c += (std::abs(s) >= std::abs(A[size_t(i) * size_t(n) + size_t(j)] * x[size_t(j)]))
                     ? ((s - t) + A[size_t(i) * size_t(n) + size_t(j)] * x[size_t(j)])
                     : ((A[size_t(i) * size_t(n) + size_t(j)] * x[size_t(j)] - t) + s);
            s = t;
            amax_row += std::abs(A[size_t(i) * size_t(n) + size_t(j)]);
        }
        s += c;
        rmax = std::max(rmax, std::abs(s - b[size_t(i)]));
        amax = std::max(amax, amax_row);
    }
    double xmax = inf_norm(x);
    double resid = rmax / (amax * xmax + 1e-300);
    if (residual_out) *residual_out = resid;
    return x;
}

Outcome<std::vector<double>> cg(const std::vector<double>& A, const std::vector<double>& b,
                                int64_t n, double tol, int64_t max_iter,
                                int64_t* iters_out, double* residual_out) {
    if (n <= 0 || int64_t(A.size()) != n * n || int64_t(b.size()) != n)
        return Status::invalid("linalg::cg: dimension mismatch");
    const size_t ns = size_t(n);
    std::vector<double> x(ns, 0.0), r = b, p = b, Ap(ns);
    double rr = 0.0;
    for (double v : r) rr += v * v;
    double bnorm = std::sqrt(rr);
    if (bnorm == 0.0) {
        if (iters_out) *iters_out = 0;
        if (residual_out) *residual_out = 0.0;
        return x;
    }
    int64_t k = 0;
    for (; k < max_iter; ++k) {
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += A[size_t(i) * size_t(n) + size_t(j)] * p[size_t(j)];
            Ap[size_t(i)] = s;
        }
        double pAp = 0.0;
        for (int64_t i = 0; i < n; ++i) pAp += p[size_t(i)] * Ap[size_t(i)];
        if (pAp <= 0.0)
            return Outcome<std::vector<double>>(
                Status::rejected("linalg::cg: matrix is not positive definite (p'Ap <= 0)"), x);
        double alpha = rr / pAp;
        for (int64_t i = 0; i < n; ++i) {
            x[size_t(i)] += alpha * p[size_t(i)];
            r[size_t(i)] -= alpha * Ap[size_t(i)];
        }
        double rr_new = 0.0;
        for (double v : r) rr_new += v * v;
        if (std::sqrt(rr_new) / bnorm <= tol) { ++k; break; }
        double beta = rr_new / rr;
        rr = rr_new;
        for (int64_t i = 0; i < n; ++i) p[size_t(i)] = r[size_t(i)] + beta * p[size_t(i)];
    }
    if (iters_out) *iters_out = k;
    if (residual_out) {
        // True residual, recomputed (never trust the recursively-updated one).
        double rmax = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += A[size_t(i) * size_t(n) + size_t(j)] * x[size_t(j)];
            rmax = std::max(rmax, std::abs(s - b[size_t(i)]));
        }
        *residual_out = rmax / (inf_norm(A) * inf_norm(x) + 1e-300);
    }
    return x;
}

std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B,
                           int64_t n, int64_t k, int64_t m) {
    std::vector<double> C(size_t(n) * size_t(m), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const double a = A[size_t(i) * size_t(k) + size_t(p)];
            if (a == 0.0) continue;
            for (int64_t j = 0; j < m; ++j)
                C[size_t(i) * size_t(m) + size_t(j)] += a * B[size_t(p) * size_t(m) + size_t(j)];
        }
    }
    return C;
}

// Exact 1-norm condition number: cond_1(A) = ||A||_1 * ||A^-1||_1, computed by
// n LU back-substitutions. Cost O(n^3), exact (up to rounding), and small enough
// for every matrix size this engine works with.
double condition_1norm(const std::vector<double>& A, int64_t n) {
    std::vector<double> LU = A;
    std::vector<int64_t> piv;
    int64_t sign = 1;
    if (!lu_factor(LU, n, piv, &sign)) return std::numeric_limits<double>::infinity();
    double anorm = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += std::abs(A[size_t(i) * size_t(n) + size_t(j)]);
        anorm = std::max(anorm, s);
    }
    double invnorm = 0.0;
    std::vector<double> e(size_t(n), 0.0);
    for (int64_t j = 0; j < n; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[size_t(j)] = 1.0;
        auto col = lu_solve_factored(LU, n, piv, e);
        if (int64_t(col.size()) != n) return std::numeric_limits<double>::infinity();
        double s = 0.0;
        for (double v : col) s += std::abs(v);
        invnorm = std::max(invnorm, s);
    }
    return anorm * invnorm;
}

double condition_estimate(const std::vector<double>& A, int64_t n) {
    // Higham's 1-norm estimator driven through the LU factors (a few iterations).
    std::vector<double> LU = A;
    std::vector<int64_t> piv;
    int64_t sign = 1;
    if (!lu_factor(LU, n, piv, &sign)) return std::numeric_limits<double>::infinity();
    const size_t ns = size_t(n);
    std::vector<double> e(ns, 1.0 / double(n)), x, w(ns);
    double est = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        x = lu_solve_factored(LU, n, piv, e);
        if (int64_t(x.size()) != n) break;
        // w = A^T sign(x) ... applied via exact transpose of the original matrix.
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                double xi = x[size_t(i)];
                s += A[size_t(i) * size_t(n) + size_t(j)] * (xi >= 0 ? 1.0 : -1.0);
            }
            w[size_t(j)] = s;
        }
        double wmax = 0.0;
        int64_t idx = 0;
        for (int64_t j = 0; j < n; ++j)
            if (std::abs(w[size_t(j)]) > wmax) { wmax = std::abs(w[size_t(j)]); idx = j; }
        if (wmax <= 0.0) break;
        double x1 = 0.0;
        for (double v : x) x1 += std::abs(v);
        est = std::max(est, x1);
        std::fill(e.begin(), e.end(), 0.0);
        e[size_t(idx)] = 1.0;
        (void)x;
    }
    double anorm = 0.0;
    for (int64_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += std::abs(A[size_t(i) * size_t(n) + size_t(j)]);
        anorm = std::max(anorm, s);
    }
    return est * anorm;
}

}  // namespace oct::numerics::linalg
