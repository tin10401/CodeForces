#include <bits/stdc++.h>
using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
int main(int argc, char* argv[]) {
    long long N = 2000;
    uniform_int_distribution<int> dist_n(1, N);
    uniform_int_distribution<int> dist_q(1, N);
    uniform_int_distribution<int> A(1, 1e9);
    uniform_int_distribution<int> val(0, 1);
    vector<int> v = {1, 0};
    int n = dist_n(rng);
    cout << n << endl;
    for(int i = 0; i < n; i++) {
        cout << v[val(rng)];
    }
    cout << endl;
    return 0;
}

