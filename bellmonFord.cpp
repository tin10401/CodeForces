#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

void solve()
{
    int n, m;
    cin >> n >> m;
    vector<array<int, 3>> graph;
    vector<int> dp(n + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b, c;
        cin >> a >> b >> c;
        graph.push_back({a, b, c});
    }
    int x;
    vector<int> parent(n + 1, -1);
    for(int i = 0; i < n; i++)
    {
        x = -1;
        for(auto& [u, v, w] : graph) 
        {
            if(w + dp[u] < dp[v])
            {
                parent[v] = u;
                x = v;
                dp[v] = dp[u] + w;
            }
        }
    }
    if(x == -1) cout << "IMPOSSIBLE" << endl;
    else
    {
        cout << yes;
        for(int i = 0; i < n; i++) x = parent[x];
        vector<int> cycle;
        for(int i = x; ; i = parent[i])
        {
            cycle.push_back(i);
            if(i == x && cycle.size() > 1) break;
        }
        reverse(all(cycle));
        for(auto& it : cycle) cout << it << " ";
        cout << endl;
    }
    
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

