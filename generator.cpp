#include <bits/stdc++.h>
using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
int main(int argc, char* argv[]) {
    int N = 200000;
    int x = 1e9;
    uniform_int_distribution<int> dist_n(N, N);
    uniform_int_distribution<int> dist_q(1, N);
    uniform_int_distribution<int> A(1, 10);
    uniform_int_distribution<int> val(x, x);

    int n = dist_n(rng);
    cout << n << endl;
    for(int i = 0; i < n / 2; i++) {
        cout << x << ' ';
    }
    for(int i = n / 2; i < n; i++) {
        cout << 0 << (i == n - 1 ? '\n' : ' ');
    }
    
    return 0;
}

