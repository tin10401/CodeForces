#include <bits/stdc++.h>
using namespace std;

int main() {
    std::mt19937_64 rng(std::random_device{}());
    const int T = 1;
    const int MAX_N = 10;
    const int MAX_M = 5;
    const int MAX_Q = 20;

    cout << T << "\n";
    for(int tc = 0; tc < T; tc++) {
        int n = std::uniform_int_distribution<int>(1, MAX_N)(rng);
        int m = std::uniform_int_distribution<int>(1, MAX_M)(rng);
        cout << n << " " << m << "\n";
        for(int i = 0; i < m; i++) {
            int x = std::uniform_int_distribution<int>(0, 1)(rng);
            int a = std::uniform_int_distribution<int>(1, n)(rng);
            int b = std::uniform_int_distribution<int>(a, n)(rng);
            cout << x << " " << a << " " << b << "\n";
        }
        int q = std::uniform_int_distribution<int>(1, MAX_Q)(rng);
        cout << q << "\n";
        for(int i = 0; i < q; i++) {
            int l = std::uniform_int_distribution<int>(1, m)(rng);
            int r = std::uniform_int_distribution<int>(l, m)(rng);
            cout << l << " " << r << "\n";
        }
    }
    return 0;
}

