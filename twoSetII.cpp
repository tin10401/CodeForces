#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll unsigned long long
#define int long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(ll base, ll exp)
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


void solve()
{
    int n;
    cin >> n;
    int sum = n * (n + 1) / 2;
    if(sum & 1)
    {
        cout << 0 << endl;
        return;
    }
    sum /= 2;
    vector<vector<int>> dp(n + 1, vector<int>(sum + 1));
    dp[0][0] = 1;
    for(int s = 0; s <= sum; s++)
    {
        for(int i = 1; i < n; i++)
        {
            dp[i][s] = dp[i - 1][s];
            if(s >= i) dp[i][s] = (dp[i][s] + dp[i - 1][s - i]) % mod;
        }
    }
    cout << dp[n - 1][sum] << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

