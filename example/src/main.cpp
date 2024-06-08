#include <iostream>
#include <RationalFitter.h>

using namespace minialg;
int main()
{
    // Generate 2D  data
    const size_t n = 11;
    const double Dtheta = M_PI / (n - 1);
    std::vector<mini::Vector<double, 2>> points;
    std::vector<double> angles;
    for (size_t i = 0; i < n; i++)
    {
        const double theta = i * Dtheta;
        angles.push_back(theta);
        points.emplace_back(std::array<double, 2>{10 * exp(-theta * 0.5) * cos(theta), 5 * sin(theta)});
    }

    // A. Test pure regression
    ratfit::RationalFitter<2> rat_pure_regression(                                                                //
        std::vector<double>{angles[0], angles[1], angles[2], angles[3], angles[4], angles[5]},                    //
        std::vector<mini::Vector<double, 2>>({points[0], points[1], points[2], points[3], points[4], points[5]}), //
        std::vector<size_t>(),                                                                                    // no indexes fixed
        std::vector<double>({1, 1, 1, 1, 1, 1}));

    std::cout << "REGRESSION\n=======================\n";
    for (size_t i = 0; i < 6; i++)
    {
        const double theta = angles[i];
        mini::Vector<double, 2> gt = points[i]; // gt[0] = cos(theta); gt[1] = sin(theta);
        mini::Vector<double, 2> prediction = rat_pure_regression.Value(theta);

        std::cout << " y(" << theta << ") = " << gt << " and prediction  f(" << theta << ") = "
                  << prediction << "\nerror: " << mini::Norm2((prediction - gt)) << std::endl;
    }

    // B. Test regression with 3 fixed points
    ratfit::RationalFitter<2> rat_regression_interp(                                                              //
        std::vector<double>{angles[0], angles[1], angles[2], angles[3], angles[4], angles[5]},                    //
        std::vector<mini::Vector<double, 2>>({points[0], points[1], points[2], points[3], points[4], points[5]}), //
        std::vector<size_t>({0, 1, 2, 3}),                                                                        // no indexes fixed
        std::vector<double>({1, 1, 1, 1, 1, 1}));

    std::cout << "REGRESSION WITH FIXED POINTS\n=======================\n";
    for (size_t i = 0; i < 6; i++)
    {
        const double theta = angles[i];
        mini::Vector<double, 2> gt = points[i]; // gt[0] = cos(theta); gt[1] = sin(theta);
        mini::Vector<double, 2> prediction = rat_regression_interp.Value(theta);

        std::cout << " y(" << theta << ") = " << gt << " and prediction  f(" << theta << ") = "
                  << prediction << "\nerror: " << mini::Norm2((prediction - gt)) << std::endl;
    }

    return 0;
}
