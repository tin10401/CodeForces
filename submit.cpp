#include <bits/stdc++.h>
using namespace std;

const int MOD = 1'000'000'007;

int N, K;
// memo[i][s][j] = #ways after processing i indices, current oddness s, with j open pairs
vector<vector<vector<int>>> memo;

int solve(int i, int s, int j) {
    if (s > K) return 0;
    if (i == N) return (s == K && j == 0) ? 1 : 0;

    int &ret = memo[i][s][j];
    if (ret != -1) return ret;

    long long res = 0;
    int s2 = s + 2 * j;              // crossing the top and bottom cuts adds 2*j to oddness
    if (s2 <= K) {
        // (1) new top -> new bottom (pair together): j unchanged, 1 way
        res += solve(i + 1, s2, j);

        // (2) both to the right (create one top-open and one bottom-open): j -> j+1, 1 way
        if (j + 1 <= N) res += solve(i + 1, s2, j + 1);

        if (j > 0) {
            // (3) mixed: close one open and create one open (j unchanged), 2*j ways
            res = (res + (2LL * j % MOD) * solve(i + 1, s2, j)) % MOD;

            // (4) both to the left: close two opens (j -> j-1), j*j ways
            res = (res + (1LL * j * j % MOD) * solve(i + 1, s2, j - 1)) % MOD;
        }
    }

    return ret = (int)(res % MOD);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N >> K;
    memo.assign(N + 1, vector<vector<int>>(K + 1, vector<int>(N + 1, -1)));
    cout << solve(0, 0, 0) << '\n';
    return 0;
}

