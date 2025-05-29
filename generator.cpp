#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main() {
    // Constants for maximum sizes
    const int MAXN = 100;
    const int MAXA = 20;

    // high‐quality RNG seeded from random_device
    mt19937 rng(random_device{}());
    uniform_int_distribution<int> distN(1, MAXN);
    uniform_int_distribution<int> distA(0, MAXA);

    // pick N and generate the array
    int N = distN(rng);
    cout << N << "\n";
    for (int i = 0; i < N; i++) {
        cout << distA(rng) << (i+1 == N ? '\n' : ' ');
    }
    return 0;
}

