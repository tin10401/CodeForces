#include <bits/stdc++.h>
using namespace std;
#define INF 1e18
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
using ll = long long;
using pll = pair<ll, ll>;
using vll = vector<ll>;

void solve() {
    int n, m; cin >> n >> m;
    int src = 0, dest = 0;
    vector<set<int>> g(n);
    for(int i = 0; i < m; i++) {
        int b, p; cin >> b >> p;
        g[b].insert(p);
        if(i == 0) src = b;
        else if(i == 1) dest = b;
    }
    priority_queue<pll, vector<pll>, greater<pll>> q;
    vll dp(n, INF);
    auto process = [&](int node, ll cost) -> void {
        if(dp[node] > cost) {
            dp[node] = cost;
            q.push({cost, node});
        }
    };
    process(src, 0);
    while(!q.empty()) {
        auto [cost, node] = q.top(); q.pop(); 
        if(dp[node] != cost) continue;
        if(node == dest) {
            cout << cost << '\n';
            return;
        }
        for(auto& dog : g[node]) {
            for(int u = node - dog, c = 1; u >= 0; u -= dog, c++) {
                process(u, cost + c);
                if(g[u].count(dog)) break;
            }
            for(int u = node + dog, c = 1; u < n; u += dog, c++) {
                process(u, cost + c);
                if(g[u].count(dog)) break;
            }
        }
    }
    cout << -1 << '\n';
}

signed main() {
    IOS;
    solve();
    return 0;
}
