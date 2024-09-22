
class Solution {
  public:
    long long treeGame(int n, vector<int> &p, vector<int> &a) {
        // code here
        using ll = long long;
        vector<vector<ll>> dp(n, vector<ll>(4));
        vector<vector<int>> graph(n);
        for(int i = 1; i < n; i++) {
            int par = p[i - 1] - 1;
            graph[par].push_back(i);
        }
        auto dfs = [&](auto& dfs, int node = 0, int par = -1) -> void {
            for(auto& nei : graph[node]) {
                dfs(dfs, nei, node);
                dp[node][0] += dp[nei][3];
            }
            vector<ll> c = {0, (ll)-1e18};
            for(auto& nei : graph[node]) {
                auto nc = c;
                nc[0] = max(c[0] + dp[nei][0], c[1] + dp[nei][2]);
                nc[1] = max(c[1] + dp[nei][0], c[0] + dp[nei][2]);
                c = nc;
            }
            dp[node][1] = a[node] + c[1];
            dp[node][2] = a[node] + c[0];
            dp[node][3] = max(dp[node][1], dp[node][0]);
        };
        dfs(dfs);
        return dp[0][3];
    }
};

