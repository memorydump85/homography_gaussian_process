#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>


// A minimal representation of an element in the sl(3) lie algebra.
class HomographyParams {
public:
  static const double kGenerators[8][3][3];
  using Vector8d = Eigen::Matrix<double, 8, 1>;

  HomographyParams(const Vector8d &v) : params_(v) {}

  HomographyParams(const double *const param_data)
      : params_(param_data) {}

  Eigen::Matrix3d lieAlgebraElement() const {
    Eigen::Matrix3d algebra_element = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 8; ++i) {
      const auto& generator = Eigen::Map<const Eigen::Matrix3d>(&kGenerators[i][0][0]).transpose();
      algebra_element += params_(i) * generator;
    }
    return algebra_element;
  }

  Eigen::Matrix3d expMap() const {
    return lieAlgebraElement().exp();
  }

private:
  const Vector8d params_;
};

int main() {
  const auto& p = HomographyParams::Vector8d::Random();
  HomographyParams params(p);
  std::cout << params.expMap() << std::endl;
  std::cout << "Determinant: " << params.expMap().determinant() << std::endl;

  return 0;
}


const double HomographyParams::kGenerators[8][3][3] = { // clang-format off
  {{ 0.,  0., +1.},
   { 0.,  0.,  0.},
   { 0.,  0.,  0.}},

  {{ 0.,  0.,  0.},
   { 0.,  0., +1.},
   { 0.,  0.,  0.}},

  {{ 0., +1.,  0.},
   { 0.,  0.,  0.},
   { 0.,  0.,  0.}},

  {{ 0.,  0.,  0.},
   {+1.,  0.,  0.},
   { 0.,  0.,  0.}},

  {{+1.,  0.,  0.},
   { 0., -1.,  0.},
   { 0.,  0.,  0.}},

  {{ 0.,  0.,  0.},
   { 0., -1.,  0.},
   { 0.,  0., +1.}},

  {{ 0.,  0.,  0.},
   { 0.,  0.,  0.},
   {+1.,  0.,  0.}},

  {{ 0.,  0.,  0.},
   { 0.,  0.,  0.},
   { 0., +1.,  0.}}}; // clang-format on
