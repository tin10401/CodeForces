#include <bits/stdc++.h>
using namespace std;

// Terry's Outing — dual/line-graph view realized via Euler-tour alternation.
// Key points:
// 1) Add dummy node D and connect all odd-degree vertices to D.
// 2) Run one Euler tour starting from D (if D has any edges), alternating 1/2.
//    This keeps any odd-length-tour imbalance at D, not at a real vertex.
// 3) Run Euler tours on remaining components (all-even). If a component has an
//    odd number of edges, exactly one real vertex gets imbalance 2 (optimal).
// 4) Minimum cost = sum_v ceil(d_v^2 / 2) + 2 * (#components with all degrees
//    even and an odd edge count).

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T; 
    if (!(cin >> T)) return 0;
    for (int tc = 1; tc <= T; ++tc) {
        int N, M; 
        cin >> N >> M;

        vector<pair<int,int>> E(M+1);
        vector<vector<pair<int,int>>> g(N + 2); // 1..N real, N+1 = D
        vector<int> deg(N + 2, 0);

        for (int i = 1; i <= M; ++i) {
            int u, v; cin >> u >> v;
            E[i] = {u, v};
            g[u].push_back({v, i});
            g[v].push_back({u, i});
            deg[u]++; deg[v]++;
        }

        // Add dummy edges from odd-degree vertices to D
        int D = N + 1;
        int eid = M;
        for (int v = 1; v <= N; ++v) {
            if (deg[v] & 1) {
                ++eid;
                g[v].push_back({D, eid});
                g[D].push_back({v, eid});
            }
        }

        // Hierholzer stacks + alternating edge colors
        vector<int> it(N + 2, 0);
        vector<char> used(eid + 1, 0);
        string ans(M, '1');
        int flip = 0;

        auto has_unused = [&](int u) -> bool {
            for (auto &pr : g[u]) if (!used[pr.second]) return true;
            return false;
        };

        auto run = [&](int s) {
            vector<int> vs, es;
            vs.push_back(s);
            while (!vs.empty()) {
                int u = vs.back();
                while (it[u] < (int)g[u].size() && used[g[u][it[u]].second]) ++it[u];
                if (it[u] == (int)g[u].size()) {
                    vs.pop_back();
                    if (!es.empty()) {
                        int e = es.back(); es.pop_back();
                        if (e <= M) ans[e - 1] = (flip ? '2' : '1');
                        flip ^= 1;
                    }
                } else {
                    auto [v, e] = g[u][it[u]++];
                    if (used[e]) continue;
                    used[e] = 1;
                    vs.push_back(v);
                    es.push_back(e);
                }
            }
        };

        // VERY IMPORTANT: drain the big "odd-degree" component by starting at D,
        // so any odd-length-tour imbalance stays at D (not a real vertex).
        if (has_unused(D)) run(D);
        // Now handle remaining (all-even) components.
        for (int v = 1; v <= N; ++v) if (has_unused(v)) run(v);

        // Compute minimal total cost:
        // base = sum ceil(d_v^2 / 2)
        long long base = 0;
        for (int v = 1; v <= N; ++v) {
            long long d = deg[v];
            base += (d*d + 1) / 2;
        }

        // penalty = +2 for each connected component that has:
        // (i) all degrees even, and (ii) an odd number of edges.
        vector<vector<int>> g0(N + 1);
        for (int i = 1; i <= M; ++i) {
            auto [u, v] = E[i];
            g0[u].push_back(v);
            g0[v].push_back(u);
        }
        vector<char> vis(N + 1, 0);
        long long penalty = 0;
        for (int s = 1; s <= N; ++s) {
            if (vis[s] || g0[s].empty()) continue;
            // BFS this component
            long long edge_cnt = 0;
            bool all_even = true;
            queue<int> q; q.push(s); vis[s] = 1;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                edge_cnt += (int)g0[u].size();
                if (deg[u] & 1) all_even = false;
                for (int w : g0[u]) if (!vis[w]) vis[w] = 1, q.push(w);
            }
            edge_cnt /= 2;
            if (all_even && (edge_cnt & 1)) penalty += 2;
        }

        long long best = base + penalty;
        cout << "Case #" << tc << ": " << best << ' ' << ans << "\n";
    }
    return 0;
}

