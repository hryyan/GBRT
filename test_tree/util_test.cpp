#include "util_test.h"

int sort_apply_permutation_test()
{
    vector<int> vectorA;
    vector<string> vectorB;

    vectorA.push_back(2);
    vectorA.push_back(1);
    vectorA.push_back(3);
    vectorA.push_back(0);
    vectorA.push_back(10);
    vectorB.push_back("Two");
    vectorB.push_back("One");
    vectorB.push_back("Three");
    vectorB.push_back("Zero");
    vectorB.push_back("Ten");

    auto p = sort_permutation(vectorA,
        [](int const& a, int const& b){ return a<b;});

    vectorA = apply_permutation(vectorA, p);
    vectorB = apply_permutation(vectorB, p);

    for (int i = 0; i < vectorA.size(); i++)
    {
        cout << vectorA[i] << " \t" << vectorB[i] << endl;
    }
    return 0;
}
