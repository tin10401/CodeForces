#include <bits/stdc++.h>
using namespace std;

int main() {
    mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> small(1, 10);
    uniform_int_distribution<int> edgeW(1, 10);
    uniform_int_distribution<int> H(0, 10);

    int n = small(rng);
    int m = small(rng);

    cout << n << " " << m << "\n";
    // n–1 edge lengths
    for (int i = 2; i <= n; i++) {
        cout << edgeW(rng) << "\n";
    }
    // m queries
    uniform_int_distribution<int> pickNode(1, n);
    for (int i = 0; i < m; i++) {
        int a = pickNode(rng);
        int h = H(rng);
        cout << a << " " << h << "\n";
    }
    return 0;
}

