//https://cses.fi/problemset/task/1716/
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1LL << 61;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(int base, int exp, int mod)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}

int m = 2e6 + 2;
int fact[2000002] = {}, invFact[2000002] = {};
void generate()
{
    fact[0] = fact[1] = 1;
    for(int i = 2; i < m; i++) fact[i] = (fact[i - 1] * i) % MOD;
    invFact[m - 1] = modExpo(fact[m - 1], MOD - 2, MOD);
    for(int i = m - 2; i >= 0; i--) invFact[i] = (invFact[i + 1] * (i + 1)) % MOD;
}
int choose(int a, int b)
{
    int res = fact[a];
    res = (res * invFact[b]) % MOD;
    res = (res * invFact[a - b]) % MOD;
    return res;
}
void solve()
{
    generate();
    int n, m;
    cin >> n >> m;
    cout << choose(n + m - 1, m) << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

