# 5 DoF Regression & Interpolation with a Rational Quadratic Model

Implementation of the algorithm published in the *ICRA24* paper, ["Efficient Pose Prediction with Rational Regression applied to vSLAM" by G. Terzakis and M. Lourakis](https://www.researchgate.net/publication/379443427_Efficient_Pose_Prediction_with_Rational_Regression_applied_to_vSLAM#fullTextFileContent).

Note that the repository merely contains the actual regression algorithm for arbitrary data dimensionality and does not include any special pose handling representations and/or algorithms (e.g., quaternions, orientation parametrization, etc.). Thus, the data employed in the example are not drawn from an actual motion trajectory. Instead, they are simple 2D points on a circle.

### Implementation details

Since the method requires only small matrix algebra, the code does not depend on any external library. It uses a custom header file for small matrix operations (`minialg.h`) and the necessary eigen decompositions are performed analytically, also by a header-only small library (`eigen3x4`). The actual method is implemented in `RationalFitter.h`.

#### Exporting/importing data from other algebra libraries

The regression class is constructed with `minialg` matrices, which can be easily constructed/ported from/to structures from other libraries such as `cv::Matx` or `Eigen::Matrix`.

### Building the example

Create a `build` directory in the repository root and execute,

`cd build`

then,

`cmake ..`

followed by,

`make`

This should build the example executable, `ratfit_example`, which fits the rational quadratic to points on a 2D circle. Note that the example runs regression twice, once for simple regression and one more time with fixed data points. 

### Cite as
If you use this code in your published work, please cite the following [paper](https://www.researchgate.net/publication/379443427_Efficient_Pose_Prediction_with_Rational_Regression_applied_to_vSLAM#fullTextFileContent):
<br>
<pre>
@inproceedings{terzakis2024rational,
  title={Efficient Pose Prediction with Rational Regression Applied to {vSLAM}},
  author={George Terzakis and Manolis Lourakis},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  pages={},
  year={2024},
  publisher={IEEE}
}
<pre>
