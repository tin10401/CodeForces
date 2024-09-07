//https://cses.fi/problemset/task/1653/
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
    int n, x;
    cin >> n >> x;
    vector<int> arr(n);
    for(auto& it : arr) cin >> it;
    pair<int, int> dp[1 << n];
    dp[0] = {0, x + 1}; 
    for(int mask = 1; mask < 1 << n; mask++)
    {
        dp[mask] = {26, 0};
        for(int i = 0; i < n; i++)
        {
            if((mask >> i) & 1)
            {
                auto [c, w] = dp[mask ^ (1 << i)];
                if(w + arr[i] > x)
                {
                    c++;
                    w = min(w, arr[i]);
                }
                else w += arr[i];
                if(dp[mask].first > c || dp[mask].first == c && dp[mask].second > w) dp[mask] = {c, w};
            }
        }
    }
    cout << dp[(1 << n) - 1].first << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

