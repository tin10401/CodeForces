ll maxPerimeter(const vvi& grid) { // max_rectangle in a grid
    int n = grid.size(), m = grid[0].size();
    ll best = 0;
    vi heights(m, 0), L(m), R(m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j)
            heights[j] = (grid[i][j] == 0 ? heights[j] + 1 : 0);
        stack<int> st;
        for (int j = 0; j < m; ++j) {
            while (!st.empty() && heights[st.top()] >= heights[j])
                st.pop();
            L[j] = st.empty() ? -1 : st.top();
            st.push(j);
        }
        while (!st.empty()) st.pop();
        for (int j = m - 1; j >= 0; --j) {
            while (!st.empty() && heights[st.top()] >= heights[j])
                st.pop();
            R[j] = st.empty() ? m : st.top();
            st.push(j);
        }
        for (int j = 0; j < m; ++j) {
            int h = heights[j];
            if (h == 0) continue;
            int w = R[j] - L[j] - 1;
            best = max(best, 2LL * (h + w));
        }
    }
    return best;
}

//    root.apply_func = [&root](iter, pmm val) -> void { -> apply ai * x + y
//        auto& r = root.root[i];
//        auto& l = root.lazy[i];
//        r = r * val.ff + val.ss * (right - left + 1);
//        l = {l.ff * val.ff, val.ff * l.ss + val.ss};
//    };

template<typename T>
ll LIS(vt<T>a, bool strict = false) { // strictly increasing or not
    auto b(a);
    if(strict) {
        for(auto& x : a) b.pb(x - 1);
    }
    srtU(b);
    const int N = b.size();
    auto get_id = [&](T x) -> int {
        return int(lb(all(b), x) - begin(b));
    };
    FW<T> root(N, 0, [](const T& a, const T& b) {return max(a, b);});
    for(auto& x : a) {
        root.update_at(get_id(x), root.get(get_id(x - strict)) + 1);
    }
    return root.get(N - 1);
}

vi square_permutation(vi a) { // return a permutation where b[b[i]] = a[i], empty vector if not possible
    // https://codeforces.com/contest/612/problem/E
    int n = a.size();
    vi vis(n);
    vvi cycles;
    for(int i = 0; i < n; i++) {
        if(vis[i]) continue;
        vi cycle;
        int j = i;
        while(!vis[j]) {
            cycle.pb(j);
            vis[j] = true;
            j = a[j];
        }
        cycles.pb(cycle);
    }
    vi last(n + 1, -1);
    fill(all(a), -1);
    for(int i = 0; i < (int)cycles.size(); i++) {
        const int N = cycles[i].size();
        if(N & 1) {
            auto& curr = cycles[i];
            for(int j = 0; j < N; j++) {
                a[curr[j]] = curr[(j + (N + 1) / 2) % N];
            }
            continue;
        }
        if(last[N] == -1) last[N] = i;
        else {
            auto& A = cycles[i];
            auto& B = cycles[last[N]];
            last[N] = -1;
            for(int j = 0; j < N; j++) {
                a[A[j]] = B[j];
                a[B[j]] = A[(j + 1) % N];
            }
        }
    }
    if(count(all(a), -1)) return {};
    return a;
}

vpii spriral_matrix(const vvi& a, int x, int y) { // do a spriral_matrix surrounding x, y as a source
    int n = a.size();
    int m = a[0].size();
    auto in = [&](int r, int c) -> bool {
        return r >= 0 && c >= 0 && r < n && c < m;
    };
    vpii now;
    if(in(x, y)) now.pb(MP(x, y));
    int total = n * m;
    int step = 1;
    while(now.size() < total) {
        for(int i = 0; i < step && now.size() < total; i++) {
            y++;
            if(in(x, y)) now.pb(MP(x, y));
        }
        for(int i = 0; i < step && now.size() < total; i++) {
            x++;
            if(in(x, y)) now.pb(MP(x, y));
        }
        step++;
        for(int i = 0; i < step && now.size() < total; i++) {
            y--;
            if(in(x, y)) now.pb(MP(x, y));
        }
        for(int i = 0; i < step && now.size() < total; i++) {
            x--;
            if(in(x, y)) now.pb(MP(x, y));
        }
        step++;
    }
    return now;
}

template<typename T>
T lcm_mod(vi a) { // lcm of multiple number under a mod
    map<int, int> mp;
    for(auto& x : a) {
        for(auto& p : primes) {
            if(p * p > x) break;
            if(x % p) continue;
            int cnt = 0;
            while(x % p == 0) {
                cnt++;
                x /= p;
            }
            mp[p] = max(mp[p], cnt); // only max occcurences matter
        } 
        mp[x] = max(mp[x], 1);
    }
    T res = 1;
    for(auto& [x, v] : mp) {
        res *= T(x).pow(v);
    }
    return res;
}

int count_distinct_palindromic_subsequence(const string& S, int mod) { // https://leetcode.com/problems/count-different-palindromic-subsequences/description/
    int N = S.size();
    if (N == 0) return 0;
    map<char, int> mp;
    int K = 0;
    vi A(N);
    for (int i = 0; i < N; ++i) {
        char c = S[i];
        auto it = mp.find(c);
        if (it == mp.end()) {
            mp[c] = K++;
            A[i] = mp[c];
        } else {
            A[i] = it->second;
        }
    }
    vvi prv(N, vi(K, -1)), nxt(N, vi(K, -1));
    vi last(K, -1);
    for (int i = 0; i < N; ++i) {
        last[A[i]] = i;
        for (int x = 0; x < K; ++x)
            prv[i][x] = last[x];
    }
    fill(last.begin(), last.end(), -1);
    for (int i = N - 1; i >= 0; --i) {
        last[A[i]] = i;
        for (int x = 0; x < K; ++x)
            nxt[i][x] = last[x];
    }
    vvi memo(N, vi(N, -1));
    auto dfs = [&](auto& dfs, int i, int j) -> int {
        if (i > j) return 1;
        if (memo[i][j] != -1) return memo[i][j];
        ll ans = 1;
        for(int x = 0; x < K; ++x) {
            int i0 = nxt[i][x], j0 = prv[j][x];
            if(i0 != -1 && i0 <= j) ans = (ans + 1) % mod;
            if(i0 != -1 && j0 != -1 && i0 < j0) ans = (ans + dfs(dfs, i0 + 1, j0 - 1)) % mod;
        }
        return memo[i][j] = ans;
    };
    int result = dfs(dfs, 0, N - 1) - 1;
    if (result < 0) result += mod;
    return result;
}

int count_assignment(int n, const vpii& edges) { // count the number of way to assign edge to vertex without having node with degree >= 2
    // https://codeforces.com/contest/2098/problem/D
    // https://codeforces.com/problemset/problem/859/E
    struct DSU { 
        public: 
            int n, comp;  
            vi root, rank, col, self_loop, edges; 
            bool is_bipartite;  
            DSU(int n) {    
                this->n = n;    
                comp = n;
                root.rsz(n, -1), rank.rsz(n, 1), col.rsz(n, 0), self_loop.rsz(n), edges.rsz(n);
                is_bipartite = true;
            }

            int find(int x) {   
                if(root[x] == -1) return x; 
                int p = find(root[x]);
                col[x] ^= col[root[x]];
                return root[x] = p;
            }

            bool merge(int u, int v) {  
                u = find(u), v = find(v);   
                if(u == v) {    
                    if(col[u] == col[v]) 
                        is_bipartite = false;
                    return false;
                }
                if(rank[v] > rank[u]) swap(u, v); 
                comp--;
                self_loop[u] |= self_loop[v];
                edges[u] += edges[v];
                root[v] = u;
                col[v] = col[u] ^ col[v] ^ 1;
                rank[u] += rank[v];
                return true;
            }

            bool same(int u, int v) {    
                return find(u) == find(v);
            }

            void assign_self_loop(int x) {
                self_loop[find(x)] = true;
            }

            void increment_edges(int x) {
                edges[find(x)]++;
            }

            int get_rank(int x) {    
                return rank[find(x)];
            }

            vvi get_group() {
                vvi ans(n);
                for(int i = 0; i < n; i++) {
                    ans[find(i)].pb(i);
                }
                return ans;
            }
    };
    DSU root(n);
    for(auto& [u, v] : edges) {
        root.merge(u, v); 
        root.increment_edges(u); // add one more edge to the component
        if(u == v) root.assign_self_loop(u); // detect self loop
    }
    mint ans = 1;
    for(int i = 0; i < n; i++) {
        if(root.find(i) == i) {
            int v = root.get_rank(i);
            int e = root.edges[i];
            if(e > v) { // impossible
                return 0;
            }
            if(e == v - 1) {
                // it's a tree, so 1 node will have degree of 0
                // for that to happen, you have to directed all the edge outward from this edge
                // so the answer is just # of vertices
                ans *= v;
                continue;
            }
            if(v == e) {
                // exactly one cycle in this edge
                // you have to directed all the edges outside the cycle to go outward from this cycle
                // there's only one way to do that, but you can reverse the edge in the cycle to make another case
                // so a->b->c->a, anotherway is a<-b<-c<-
                // so answer is 2 if the cycle has len > 1
                // and 1 if the component has a self loop
                ans *= root.self_loop[i] ? 1 : 2;
            }
        }
    }
    return (int)ans;
}

ll kadane_2d(vvi& a) { // max subarray in 2d matrix
	// https://www.naukri.com/code360/problems/max-submatrix_1214973?leftPanelTabValue=SUBMISSION
    int n = a.size();
    int m = a[0].size();
    ll res = -INF;
    for(int top = 0; top < n; top++) {
        vll t(m);
        for(int bot = top; bot >= 0; bot--) {
            for(int j = 0; j < m; j++) {
                t[j] += a[bot][j];
            } 
            ll curr = 0;
            for(int j = 0; j < m; j++) {
                curr = max(t[j], curr + t[j]);
                res = max(res, curr);
            }
        }
    }
    return res;
}

pair<vi,vi> find_longest_cycle_bidirected_graph(const vvpii& graph){ // return a cycle and edges_id which it used
    int n = graph.size();
    vb visited(n, false), inStack(n, false);
    vi depth(n, 0), parent(n, -1), parentEdge(n, -1);
    int bestLen = 0, bestU = -1, bestV = -1, bestEdge = -1;
    auto dfs = [&](auto&& dfs, int u) -> void {
        visited[u] = inStack[u] = true;
        for (auto [v, eid] : graph[u]) {
            if (eid == parentEdge[u]) continue;
            if (!visited[v]) {
                parent[v] = u;
                parentEdge[v] = eid;
                depth[v] = depth[u] + 1;
                dfs(dfs, v);
            }
            else if (inStack[v]) {
                int len = depth[u] - depth[v] + 1;
                if (len > bestLen) {
                    bestLen = len;
                    bestU = u;
                    bestV = v;
                    bestEdge = eid;
                }
            }
        }
        inStack[u] = false;
    };

    for (int i = 0; i < n; i++)
        if (!visited[i])
            dfs(dfs, i);

    if (bestLen == 0)
        return {vi(), vi()};

    vi verts;
    int cur = bestU;
    while (cur != bestV) {
        verts.pb(cur);
        cur = parent[cur];
    }
    verts.pb(bestV);
    rev(verts);

    vi eids;
    for (int i = 1; i < (int)verts.size(); i++)
        eids.pb(parentEdge[verts[i]]);
    eids.pb(bestEdge);

    return {verts, eids};
}

ll nC2_vector(const std::vector<ll>& a) { // compute sum(a[i] * a[j]) over all pair
    ll sum   = 0;
    ll sumSq = 0;
    for (ll x : a) {
        sum   += x;
        sumSq += x * x;
    }
    // (sum^2 - sum of squares) / 2 = Σ_{i<j} a[i]*a[j]
    return (sum * sum - sumSq) / 2;
}

vi get_path(const vvi& graph, int s, int e) {
    vi path;
    auto dfs = [&](auto& dfs, int node, int par) -> bool {
        if(node == e) {
            path.pb(node);
            return true;
        } 
        for(auto& nei : graph[node]) {
            if(nei == par) continue;
            if(dfs(dfs, nei, node)) {
                path.pb(node);
                return true;
            }
        }
        return false;
    };
    dfs(dfs, s, -1);
    rev(path);
    return path;
}

bool is_symmetrical(const vvi& graph, int root = 0) {
    map<vi, int> hash_code;
    map<int, int> sym;
    int cnt = 0;
    auto dfs = [&](auto& dfs, int node = 0, int par = -1) -> int {
        vi child;
        for(auto& nei : graph[node]) {
            if(nei == par) continue;
            child.pb(dfs(dfs, nei, node));
        }
        srt(child);
        if(!hash_code.count(child)) {
            map<int, int> c;
            for(auto& it : child) c[it]++;
            bool bad = false;
            int odd = 0;
            for(auto& [x, v] : c) {
                if(v & 1) {
                    odd++;
                    bad |= !sym[x];
                }
            }
            hash_code[child] = ++cnt;
            sym[cnt] = odd < 2 && !bad;
        }
        return hash_code[child];
    };
    return sym[dfs(dfs, root)];
}

vi toposort(vvi& graph) {
    int n = graph.size();
    vi degree(n);
    for (int u = 0; u < n; u++)
        for (auto v : graph[u])
            degree[v]++;
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (degree[i] == 0)
            q.push(i);
    vi ans;
    while (!q.empty()) {
        auto i = q.front(); q.pop(); ans.pb(i);
        for (auto& j : graph[i])
            if (--degree[j] == 0)
                q.push(j);
    }
    if (ans.size() != n) return {};
    return ans;
}

vi bidirectional_cycle_vector(int n, const vvi& graph) { // return a cycle_vector to mark which node is part of the cycle
    vi inCycle(n, true);
    vi degree(n);
    queue<int> q;
    for(int i = 0; i < n; i++) {
        for(auto& j : graph[i]) degree[j]++;
    }
    for (int i = 0; i < n; i++) {
        if(degree[i] <= 1) {
            q.push(i);
            inCycle[i] = false;
        }
    }
    while(!q.empty()) {
        auto u = q.front(); q.pop();
        for(int v : graph[u]) {
            if(inCycle[v]) {
                if(--degree[v] == 1) {
                    inCycle[v] = false;
                    q.push(v);
                }
            }
        }
    }
    return inCycle;
}

ll bellman_ford(int N, int src, const var(3)& edges) {
    vll dist(N, -INF);
    dist[0] = 0;
    for(int it = 1; it <= N + 1; it++){
        bool updated = false;
        for(auto &[u, v, w] : edges){
            if(dist[u] != -INF && dist[u] + w > dist[v]) {
                dist[v] = dist[u] + w;
                updated = true;
                if(it == N) {
                    return -1;
                }
            }
        }
        if(!updated) break;
    }
    return dist.back();
}

vvi rotate90(const vvi matrix) {
    int n = matrix.size(), m = matrix[0].size();
    vvi res(m, vi(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res[j][n - 1 - i] = matrix[i][j];
    return res;
}

ll count_unique(const vi& a) { // sum of unique element over all subarray
    ll n = a.size();
    map<int, vi> mp;
    for(int i = 0; i < n; i++) mp[a[i]].pb(i);
    ll total = n * (n + 1) / 2;
    ll res = 0;
    for(auto& [_, it] : mp) {
        it.pb(n); 
        int last = -1;
        ll now = 0;
        for(auto& x : it) {
            ll d = x - last - 1;
            now += d * (d + 1) / 2;
            last = x;
        }
        res += total - now;
    }
    return res;
}

vi directional_cycle_vector(const vvi& out_graph) {
    int n = out_graph.size();
    vvi in_graph(n);
    vi in_deg(n, 0), out_deg(n, 0), in_cycle(n, true);
    for(int u = 0; u < n; u++) {
        out_deg[u] = out_graph[u].size();
        for(int v : out_graph[u]) {
            in_graph[v].pb(u);
            in_deg[v]++;
        }
    }
    queue<int> q;
    for(int i = 0; i < n; i++) {
        if(in_deg[i] == 0 || out_deg[i] == 0) {
            q.push(i);
            in_cycle[i] = false;
        }
    }
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        for(int v : out_graph[u]) {
            if(in_cycle[v]) {
                if(--in_deg[v] == 0) {
                    in_cycle[v] = false;
                    q.push(v);
                }
            }
        }
        for(int v : in_graph[u]) {
            if(in_cycle[v]) {
                if(--out_deg[v] == 0) {
                    in_cycle[v] = false;
                    q.push(v);
                }
            }
        }
    }
    return in_cycle;
}

template<typename T = int>
vi diameter_vector(const vt<vt<T>>& graph) { // return a vector where a[i] is the max diameter from the ith vertex
    int n = graph.size();
    auto bfs = [&](int src) -> vi {
        vi dp(n, -1);
        queue<int> q;
        auto process = [&](int u, int c) -> void {
            if(dp[u] == -1) {
                dp[u] = c;
                q.push(u);
            }
        };
        process(src, 0);
        while(!q.empty()) {
            int node = q.front(); q.pop();
            for(auto& nei : graph[node]) {
                process(nei, dp[node] + 1);
            }
        }
        return dp;
    };
    auto dummy = bfs(0);
    int a = max_element(all(dummy)) - begin(dummy);
    auto dp1 = bfs(a);
    int b = max_element(all(dp1)) - begin(dp1);
    auto dp2 = bfs(b);
    vi ans(n);
    for(int i = 0; i < n; i++) {
        ans[i] = max(dp1[i], dp2[i]);
    }
    return ans;
}

string validate_substring(int n, const string& t, vi a) { 
    // given a string s of len n full of '?'
    // and a vector a indicates where t is a substring of s
    // determine if it's a valid sequence of not
    string s = string(n, '?');
    srtU(a); rev(a);
    auto z = Z_Function(t);
    int m = t.size();
    for(auto& i : a) {
        i--;    
        int r = i + m;
        if(i + m > n) return "";
        for(int j = i; j < r; j++) {
            int id = j - i;
            if(s[j] == '?') {
                s[j] = t[id];
                continue;
            }
            if(z[id] < r - j) return "";
            break;
        }
    }
    return s;
}
