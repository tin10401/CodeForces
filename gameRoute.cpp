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
    int n, m;
    cin >> n >> m;
    vector<int> cnt(n + 1), degree(n + 1);
    vector<vector<int>> graph(n + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        degree[b]++;
    }
    queue<int> q;
    for(int i = 2; i <= n; i++)
    {
        if(degree[i] == 0) q.push(i);
    }
    while(!q.empty())
    {
        int node = q.front(); q.pop();
        for(auto& nei : graph[node])
        {
            if(--degree[nei] == 0 && nei != 1) q.push(nei);
        }
    }
    q.push(1);
    cnt[1] = 1;
    while(!q.empty())
    {
        int node = q.front(); q.pop();
        for(auto& nei : graph[node])
        {
            cnt[nei] = (cnt[nei] + cnt[node]) % mod;
            if(--degree[nei] == 0) q.push(nei);
        }
    }
    cout << cnt[n] << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

