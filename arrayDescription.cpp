// https://cses.fi/problemset/task/1746/
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ull unsigned long long
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

ll modExpo(ll base, ll exp)
{
    ll res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}


void solve()
{
    int n, m;
    cin >> n >> m;
    vector<int> arr(n);
    for(auto& it : arr) cin >> it;
    vector<ll> dp(m + 1);
    if(arr[0] == 0)
    {
        for(auto& it : dp) it = 1;
    }
    else dp[arr[0]] = 1;
    for(int i = 1; i < n; i++)
    {
        vector<ll> next(m + 1);
        int left = arr[i] == 0 ? 1 : arr[i];
        int right = arr[i] == 0 ? m : arr[i];
        for(int j = left; j <= right; j++)
        {
            next[j] += dp[j];
            if(j > 1) next[j] += dp[j - 1];
            if(j < m) next[j] += dp[j + 1];
            next[j] %= mod;
        }
        swap(dp, next);
    }
    ll res = 0;
    for(int i = 1; i <= m; i++) res = (res + dp[i]) % mod;
    cout << res << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

