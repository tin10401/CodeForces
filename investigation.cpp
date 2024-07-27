#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll unsigned long long
#define int long long
const int INF = 1e18;
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
    vector<vector<pair<int, int>>> graph(n + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b, c;
        cin >> a >> b >> c;
        graph[a].push_back({b, c});
    }

    vector<int> dp(n + 1, INF), ways(n + 1), maxCount(n + 1, -INF), minCount(n + 1, INF);
    dp[1] = 0, ways[1] = 1;
    priority_queue<array<int, 2>, vector<array<int, 2>>, greater<array<int, 2>>> minHeap;
    minCount[1] = maxCount[1] = 0;
    minHeap.push({0, 1});
    while(!minHeap.empty())
    {
        auto [cost, node] = minHeap.top(); minHeap.pop();
        if(dp[node] != cost) continue;
        for(auto& [nei, c] : graph[node])
        {
            int newCost = cost + c;
            if(newCost < dp[nei])
            {
                minCount[nei] = minCount[node] + 1;
                maxCount[nei] = maxCount[node] + 1;
                dp[nei] = newCost;
                ways[nei] = ways[node];
                minHeap.push({newCost, nei});
            }
            else if(newCost == dp[nei])
            {
                minCount[nei] = min(minCount[nei], minCount[node] + 1);
                maxCount[nei] = max(maxCount[nei], maxCount[node] + 1);
                ways[nei] = (ways[nei] + ways[node]) % mod;
            }
        }
    }

    cout << dp[n] << " " << ways[n] << " " << minCount[n] << " " << maxCount[n] << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

