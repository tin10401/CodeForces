#pragma GCC target("popcnt")
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

const int MXN = 3005;
vi dp_curr(MXN), dp_prev(MXN);
int prefix[MXN];

int compute(int l, int r) { return (prefix[r] - prefix[l - 1]) * (prefix[r] - prefix[l - 1]);};

void dfs(int left, int right, int x, int y)
{
    if(left > right) return;
    int m = left + (right - left) / 2;
    pii p = {INF, -1};
    for(int i = x; i <= min(m, y); i++)
    {
        p = min(p, {dp_prev[i] + compute(i + 1, m), i});
    }
    dp_curr[m] = p.f;
    dfs(left, m - 1, x, p.s);
    dfs(m + 1, right, p.s, y);
}

void solve()
{
    int n, k; cin >> n >> k;
    vi arr(n);
    for(int i = 0; i < n; i++)
    {
        int val; cin >> val;
        prefix[i + 1] = prefix[i] + val;
    }
    for(int i = 1; i <= n; i++) dp_prev[i] = compute(1, i);
    for(int i = 2; i <= k; i++)
    {
        dfs(1, n, 1, n);
        dp_prev = dp_curr;
    }
    cout << dp_prev[n] << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

