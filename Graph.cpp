template<typename T, typename F = function<T(const T&, const T&)>>
class SparseTable {
public:
    int n, m;
    vt<vt<T>> st;
    vi log_table;
    F func;
    
    SparseTable() {}

    SparseTable(const vt<T>& a, F func) : n(a.size()), func(func) {
        m = floor(log2(n)) + 1;
        st.rsz(m);
        for (int j = 0; j < m; j++) st[j].rsz(n - (1 << j) + 1);
        log_table.rsz(n + 1);
        for (int i = 2; i <= n; i++) log_table[i] = log_table[i / 2] + 1;
        for (int i = 0; i < n; i++) st[0][i] = a[i];
        for (int j = 1; j < m; j++) {
            for (int i = 0; i + (1 << j) <= n; i++)
                st[j][i] = func(st[j - 1][i], st[j - 1][i + (1 << (j - 1))]);
        }
    }
    
    T query(int L, int R) {
        int j = log_table[R - L + 1];
        return func(st[j][L], st[j][R - (1 << j) + 1]);
    }
};

template<typename T = int>
struct LCA_O1 {
    vi enter;
    vpii euler;
    SparseTable<pii> st;
    int timer;
    LCA_O1() {}
    LCA_O1(const vt<vt<T>> &graph, int root = 0) : timer(0) {
        int n = graph.size();
        enter.resize(n, -1);
        dfs(root, -1, 0, graph);
        st = SparseTable<pii>(euler, [](const pii &a, const pii &b) {
            return (a.first < b.first) ? a : b;
        });
        vpii().swap(euler);
    }
    void dfs(int node, int par, int d, const vt<vt<T>> &graph) {
        enter[node] = timer++;
        euler.pb({d, node});
        for(auto& nxt : graph[node]) {
            if(nxt == par) continue;
            dfs(nxt, node, d + 1, graph);
            euler.pb({d, node});
            timer++;
        }
    }
    int lca(int u, int v) {
        int L = min(enter[u], enter[v]);
        int R = max(enter[u], enter[v]);
        return st.query(L, R).second;
    }
};

template<typename T = int>
class GRAPH { 
    public: 
    int n, m; 
    vvi dp;
    vi depth, parent, subtree;
    vi tin, tout, low, ord;
    int timer = 0;
    LCA_O1<T> lca_01;
//    int centroid1 = -1, centroid2 = -1, mn = inf, diameter = 0;
    GRAPH() {}

    GRAPH(const vt<vt<T>>& graph, int root = 0) : lca_01(graph, root) {   
        n = graph.size();
        m = log2(n) + 1;
        dp.rsz(n, vi(m));
        depth.rsz(n);
        parent.rsz(n, -1);
		subtree.rsz(n, 1);
        tin.rsz(n);
        tout.rsz(n);
		ord.rsz(n);
//        low.rsz(n);
        dfs(graph, root);
        init();
    }
    
    void dfs(const vt<vt<T>>& graph, int node = 0, int par = -1) {   
		tin[node] = timer++;
		ord[tin[node]] = node;
        for(auto& nei : graph[node]) {  
            if(nei == par) continue;    
            depth[nei] = depth[node] + 1;   
            dp[nei][0] = node;
            parent[nei] = node;
			dfs(graph, nei, node);
			subtree[node] += subtree[nei];
        }
		tout[node] = timer - 1;
//        tin[node] = timer++;
//		subtree[node] = 1;
//        int mx = 0, a = 0, b = 0;
//        for(auto& nei : graph[node]) {  
//            if(nei == par) continue;    
//            depth[nei] = depth[node] + 1;   
//            dp[nei][0] = node;
//            parent[nei] = node;
//			int v = dfs(nei, node);
//			subtree[node] += subtree[nei];
//            if(v > a) b = a, a = v; 
//            else b = max(b, v);
//			mx = max(mx, subtree[nei]);
//        }
//		diameter = max(diameter, a + b); // might be offset by 1
//        tout[node] = timer - 1;
//        mx = max(mx, n - subtree[node] - 1); // careful with offSet, may take off -1
//		if(mx < mn) mn = mx, centroid1 = node, centroid2 = -1;
//		else if(mx == mn) centroid2 = node;
//		return a + 1;
    }

//    void online_init(int u, int par, int x) {
//        depth[u] = depth[par] + 1;
//        dp[u][0] = par;
//        ans[u][0] = Node(x);
//        for(int j = 1; j < m; j++) {
//            int p = dp[u][j - 1];
//            dp[u][j] = dp[p][j - 1];
//            ans[u][j] = merge(ans[u][j - 1], ans[p][j - 1]);
//        }
//    }

    bool isAncestor(int u, int v) { 
        return tin[u] <= tin[v] && tin[v] <= tout[u]; 
    }
    
    void init() {  
        for(int j = 1; j < m; j++) {   
            for(int i = 0; i < n; i++) {    
                dp[i][j] = dp[dp[i][j - 1]][j - 1];
            }
        }
    }
	
    int lca(int a, int b) { 
        return lca_01.lca(a, b);
    }
	
	int dist(int u, int v) {    
        int a = lca_01.lca(u, v);  
        return depth[u] + depth[v] - 2 * depth[a];
    }

	int k_ancestor(int a, int k) {
        for(int i = m - 1; i >= 0; i--) {   
            if((k >> i) & 1) a = dp[a][i];
        }
        return a;
    }

    int rooted_lca(int a, int b, int c) { // determine if 3 points are in the same path
        return lca(a, c) ^ lca(a, b) ^ lca(b, c);
    }

    int rooted_parent(int u, int v) { // move one level down from u closer to v
        return k_ancestor(v, depth[v] - depth[u] - 1);
    }

    void reroot(int root) {
        fill(all(parent), -1);
        dfs(root);
        init();
    }

//    void bridge_dfs(int node = 0, int par = -1) {
//        low[node] = tin[node] = timer++; 
//        subtree[node] = 1;
//        for(auto& nei : graph[node]) {  
//            if(nei == par) continue;
//            if(!tin[nei]) {   
//                bridge_dfs(nei, node);
//                subtree[node] += subtree[nei];
//                low[node] = min(low[node], low[nei]);   
//                if(low[nei] > tin[node]) {  
//                    //res = max(res, (ll)subtree[nei] * (n - subtree[nei]));
//                }
//            }
//            else {  
//                low[node] = min(low[node], tin[nei]);
//            }
//        }
//    };
};

template<class T, typename TT = int, typename F = function<T(const T&, const T&)>>
class HLD {
    private:
	vpii get_path_helper(int node, int par) {
        vpii seg;
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                seg.pb({id[tp[node]], id[node]});
                node = parent[tp[node]];
            } else {   
                seg.pb({id[par] + 1, id[node]});
                break;  
            } 
        }   
        seg.pb({id[par], id[par]});
        return seg;
    }

	T path_queries_helper(int node, int par) { // only query up to parent, don't include parent info
        T res = DEFAULT;
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                T t = seg.queries_range(id[tp[node]], id[node]);
                res = func(t, res);
                node = parent[tp[node]];
            } else {   
                T t = seg.queries_range(id[par] + 1, id[node]);
                res = func(t, res);
                break;  
            } 
        }   
        return res; 
    }

	void update_path_helper(int node, int par, T val) {
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                seg.update_range(id[tp[node]], id[node], val);
                node = parent[tp[node]];
            } else {   
                seg.update_range(id[par] + 1, id[node], val); 
                break;  
            } 
        }   
    }
    public:
    FW<T> seg;
    vi id, tp, sz, parent;
    int ct;
    vt<vt<T>> graph;
    int n;
    GRAPH<TT> g;
    T DEFAULT;
    F func;
    HLD() {}

    HLD(vt<vt<TT>>& graph, vi a, F func, int root = 0, T DEFAULT = 0) : seg(graph.size(), DEFAULT, func), g(graph, root), n(graph.size()), func(func), DEFAULT(DEFAULT) {
        this->parent = move(g.parent);
        this->sz = move(g.subtree);
        ct = 0;
        id.rsz(n), tp.rsz(n);
        dfs(graph);
        for(int i = 0; i < n; i++) seg.update_at(id[i], a[i]);
    }
        
    void dfs(const vt<vt<TT>>& graph, int node = 0, int par = -1, int top = 0) {   
        id[node] = ct++;    
        tp[node] = top;
        int nxt = -1, max_size = -1;    
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            if(sz[nei] > max_size) {   
                max_size = sz[nei]; 
                nxt = nei;  
            }   
        }   
        if(nxt == -1) return;   
        dfs(graph, nxt, node, top);   
        for(auto& nei : graph[node]) {   
            if(nei != par && nei != nxt) dfs(graph, nei, node, nei);  
        }   
    }

    void update_at(int i, T v) {
        seg.update_at(id[i], v);
    }
	
	void update_subtree(int i, T v) {
        seg.update_range(id[i], id[i] + sz[i] - 1, v);
    }

	vpii get_path(int u, int v) {
        int p = g.lca(u, v);
        auto path = get_path(u, p);
        auto other = get_path(v, p);
        other.pop_back();
        rev(other);
        path.insert(end(path), all(other));
        return path;
    }

	T path_queries(int u, int v) { // remember to include the info of parents
        int c = lca(u, v);
        T res = func(seg.queries_at(id[c]), func(path_queries_helper(u, c), path_queries_helper(v, c)));
        return res;
    }


    void update_path(int u, int v, T val) {
        int c = lca(u, v);
        update_path_helper(u, c, val);
        update_path_helper(v, c, val);
        seg.update_at(id[c], val);
    }

    int dist(int a, int b) {
        return g.dist(a, b);
    }

    int lca(int a, int b) {
        return g.lca(a, b);
    }

    bool contain_all_node(int u, int v) {
        return path_queries(u, v) == dist(u, v);
    }
};

class DSU { 
    public: 
    int n, comp;  
    vi root, rank;  
    DSU(int n) {    
        this->n = n;    
		comp = n;
        root.rsz(n, -1), rank.rsz(n, 1);
		// minimum swap to sorted by swapping i and j in permutation will be base(n - root.comp) + (same(i, j) ? -1 : 1) 
    }
    
    int find(int x) {   
        if(root[x] == -1) return x; 
        return root[x] = find(root[x]);
    }
    
    bool merge(int u, int v) {  
        u = find(u), v = find(v);   
        if(u != v) {    
            if(rank[v] > rank[u]) swap(u, v); 
			comp--;
            rank[u] += rank[v]; 
            root[v] = u;
            return true;
        }
        return false;
    }
    
    bool same(int u, int v) {    
        return find(u) == find(v);
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

vi toposort(vvi& graph, vi degree) {
    queue<int> q;
    int n = graph.size();
    for(int i = 0; i < n; i++) if(degree[i] == 0) q.push(i);
    vi ans;
    while(!q.empty()) {
        auto i = q.front(); q.pop(); ans.pb(i);
        for(auto& j : graph[i]) if(--degree[j] == 0) q.push(j);
    }
	if(ans.size() != n) return {};
    return ans;
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

struct Persistent_DSU {
	int n, version;
    vvpii parent, rank;
	Persistent_DSU(int n) {
		this->n = n; version = 0;
		parent.rsz(n); rank.rsz(n);
		for (int i = 0; i < n; i++) {
			parent[i].pb(MP(version, i));
			rank[i].pb(MP(version, 1));
		}
	}
 
	int find(int u, int ver) {
		auto [v, par] = *(ub(all(parent[u]), MP(ver + 1, -1)) - 1);
        return par != u ? find(par, ver) : par;
	}
 
	int getRank(int u, int ver) {
		u = find(u, ver);
		auto [v, sz] = *(ub(all(rank[u]), MP(ver + 1, -1)) - 1);
		return sz;
	}
 
	int merge(int u, int v, int ver) {
		u = find(u, ver), v = find(v, ver);
		if (u == v) return 0;
		version = ver;
		int szu = rank[u].back().ss;
		int szv = rank[v].back().ss;
		if (szu < szv) {swap(u, v);}
		parent[v].pb({version, u});
		int new_sz = szu + szv;
		rank[u].pb({version, new_sz});
		return version;
	}
 
	bool same(int u, int v, int ver) {
        return find(u, ver) == find(v, ver);
	}
};

class SCC {
    public:
    int n, curr_comp;
    vvi graph, revGraph;
    vi vis, comp, degree;
    stack<int> s;
 
    SCC(int n) {
        this->n = n;
        curr_comp = 0;
        graph.resize(n), revGraph.resize(n), vis.resize(n), comp.resize(n, -1), degree.rsz(n);
		// don't forget to build after adding edges
    }
 
    void add_directional_edge(int a, int b) {    
        graph[a].pb(b); 
        revGraph[b].pb(a);
    }
 
    void dfs(int node) {
        if(vis[node]) return;
        vis[node] = true;
        for(auto& nei : graph[node]) dfs(nei);
        s.push(node);
    }
 
    void dfs2(int node) {
        if(comp[node] != -1) return;
        comp[node] = curr_comp;
        for(auto& nei : revGraph[node]) dfs2(nei);
    }
 
    void build() {
        for(int i = 0; i < n; i++) dfs(i);
        while(!s.empty()) {
            int node = s.top(); s.pop();
            if(comp[node] != -1) continue;
            dfs2(node);
            curr_comp++;
        }
    }
    
    vvi compress_graph() {    
        vvi g(curr_comp);   
        for(int i = 0; i < n; i++) {    
            for(auto& j : graph[i]) {   
                if(comp[i] != comp[j]) {    
                    g[comp[i]].pb(comp[j]);
                    degree[comp[j]]++;
                }
            }
        }
        for(auto& it : g) srtU(it);
        return g;
    }
};

template<typename T>
struct CD { // centroid_decomposition
    int n, root;
    vt<vt<T>> graph;
    vi size, parent, vis;
    ll ans;
    GRAPH<T> g;
    vi best;
    CD(const vt<vt<T>>& graph) : graph(graph), n(graph.size()), g(graph), best(graph.size(), inf) {
        ans = 0;
        size.rsz(n);
        parent.rsz(n, -1);
        vis.rsz(n);
        root = init();
    }
 
    void get_size(int node, int par) { 
        size[node] = 1;
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            get_size(nei, node);
            size[node] += size[nei];
        }
    }
 
    int get_center(int node, int par, int size_of_tree) { 
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            if(size[nei] * 2 > size_of_tree) return get_center(nei, node, size_of_tree);
        }
        return node;
    }

    int get_centroid(int src) { 
        get_size(src, -1);
        int centroid = get_center(src, -1, size[src]);
        vis[centroid] = true;
        return centroid;
    }

    int mx;
    void modify(int node, int par, int depth, int delta) {
        for(auto& nei : graph[node]) {
            if(vis[nei] || nei == par) continue;
            modify(nei, node, depth + 1, delta);
        }
    }

    void cal(int node, int par, int depth) {
        for(auto& nei : graph[node]) {
            if(vis[nei] || nei == par) continue;
            cal(nei, node, depth + 1);
        }
    }
 
    int get_max_depth(int node, int par = -1, int depth = 0) {
        int max_depth = depth;
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            max_depth = max(max_depth, get_max_depth(nei, node, depth + 1));
        }
        return max_depth;
    }

    void run(int root, int par) {
        mx = get_max_depth(root, par);
        for(auto& nei : graph[root]) {
            if(vis[nei] || nei == par) continue;
            cal(nei, root, 1);
            modify(nei, root, 1, 1);
        }
    }

    int init(int root = 0, int par = -1) {
        root = get_centroid(root);
        parent[root] = par;
        run(root, par);
        for(auto&nei : graph[root]) {
            if(nei == par || vis[nei]) continue;
            init(nei, root);
        }
        return root;
    }

    void update(int node) {
        int u = node;
        while(u != -1) {
            best[u] = min(best[u], g.dist(u, node));
            u = parent[u];
        }
    }

    int queries(int node) {
        int u = node;
        int res = inf;
        while(u != -1){ 
            res = min(res, g.dist(u, node) + best[u]);
            u = parent[u];
        }
        return res;
    }
};

struct CYCLE {
    vvi graph;
    vi degree;
    int n;
    CYCLE(vvi &graph, vi& degree) : graph(graph), degree(degree) { n = graph.size(); }
 
    vi reconstruct_cycle(int u, int v, const vi &parent) {
        vi pathU, pathV;
        for (int cur = u; cur != -1; cur = parent[cur]) pathU.pb(cur);
        for (int cur = v; cur != -1; cur = parent[cur]) pathV.pb(cur);
        rev(pathU);
        rev(pathV);
        int idx = 0;
        while (idx < (int)pathU.size() && idx < (int)pathV.size() && pathU[idx] == pathV[idx]) idx++;
        idx--;
        vi cycle;
        for (int i = (int)pathU.size() - 1; i >= idx; i--) cycle.pb(pathU[i]);
        for (int i = idx + 1; i < (int)pathV.size(); i++) cycle.pb(pathV[i]);
        return cycle;
    }
 
    vi find_shortest_cycle(int s) {
        vi dis(n, inf), parent(n, -1);
        queue<int> q;
        dis[s] = 0;
        q.push(s);
        int bestShortest = inf;
        int candU_short = -1, candV_short = -1;
        while (!q.empty()){
            int u = q.front();
            q.pop();
            for (int v : graph[u]){
                if (dis[u] + 1 < dis[v]) {
                    dis[v] = dis[u] + 1;
                    parent[v] = u;
                    q.push(v);
                } else if (v != parent[u] && dis[u] != inf && dis[v] != inf) {
                    int currLength = dis[u] + dis[v] + 1;
                    if (currLength < bestShortest) {
                        bestShortest = currLength;
                        candU_short = u;
                        candV_short = v;
                    }
                }
            }
        }
        vi shortestCycle;
        if (candU_short != -1) shortestCycle = reconstruct_cycle(candU_short, candV_short, parent);
        return shortestCycle;
    }

    vi find_longest_cycle() {
        vi vis(n, 0), steps(n, -1);
        int best_len = -1;
        vi bestCycle;
        for (int i = 0; i < n; i++) {
            if (vis[i] != 0) continue;
            int cur = i, step = 0;
            vi chain;
            map<int, int> pos;
            while (cur != -1 && vis[cur] == 0) {
                vis[cur] = i + 1;
                pos[cur] = step;
                chain.pb(cur);
                step++;
                int next = -1;
                if (!graph[cur].empty()) next = graph[cur][0];
                cur = next;
            }
            if (cur != -1 && vis[cur] == i + 1) {
                int cycleStart = pos[cur];
                int cycleLen = step - cycleStart;
                if (cycleLen > best_len) {
                    best_len = cycleLen;
                    bestCycle = vi(chain.begin() + cycleStart, chain.end());
                }
            }
        }
        return bestCycle;
    }
 
    vi get_max_independent_set(int src) {
        vvi group(2);
        vi vis(n);
        auto dfs = [&](auto& dfs, int node, int p, int d) -> void {
            vis[node] = true;
            group[d].pb(node);
            for(auto& nei : graph[node]) {
                if(!vis[nei]) dfs(dfs, nei, node, d ^ 1);
            }
        };
        dfs(dfs, src, -1, 0);
        return group[0].size() > group[1].size() ? group[0] : group[1];
    }

    vi longest_cycle_path() { // return longest path where each vertex is visited once in a DIRECTED GRAPH
        queue<int> q;
        vi vis(n);
        for(int i = 0; i < n; i++) {
            if(degree[i] == 0) {
                q.push(i);
                vis[i] = true;
            }
        }
        while(!q.empty()) {
            auto node = q.front(); q.pop();
            vis[node] = true;
            for(auto& nei : graph[node]) {
                if(--degree[nei] == 0) q.push(nei);
            }
        }
        vi cycle(n);
        for(int i = 0; i < n; i++) {
            if(!vis[i]) {
                cycle[i] = true;
            }
        }
        DSU root(n);
        for(int i = 0; i < n; i++) {
            if(cycle[i]) {
                for(auto& j : graph[i]) {
                    if(cycle[j]) root.merge(i, j);
                }
            }
        }    
        vi dp(n, -1);
        vi next(n, -1);
        auto dfs = [&](auto& dfs, int node) -> int {
            auto& res = dp[node];
            if(res != -1) return res;
            if(cycle[node]) {
                return res = root.get_rank(node);
            }
            res = 1;
            for(auto& nei : graph[node]) {
                int v = dfs(dfs, nei) + 1;
                if(v > res) {
                    res = v;
                    next[node] = nei;
                }
            }
            return res;
        };
        for(int i = 0; i < n; i++) dfs(dfs, i);
        int mx = max_element(all(dp)) - begin(dp);
        vi path;
        int node = mx;
        while(node != -1) {
            path.pb(node);
            if(cycle[node]) {
                int u = node;
                while(true) {
                    int nxt = -1;
                    for(auto& nei : graph[node]) {
                        if(root.same(nei, node)) {
                            nxt = nei;
                            break;
                        }
                    }  
                    if(nxt == u) break;
                    node = nxt;
                    path.pb(node);
                } 
                break;
            }
            node = next[node];
        }
        return path;
    }
};

vi bidirectional_cycle_vector(int n, const vvi& graph) { // return a cycle_vector to mark which node is part of the cycle
    vi inCycle(n, true);
    vi degree(n);
    queue<int> q;
    for(int i = 0; i < n; i++) {
        for(auto& j : graph[i]) degree[j]++;
    }
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) {
            q.push(i);
            inCycle[i] = false;
        }
    }
    while (!q.empty()) {
        auto u = q.front(); q.pop();
        for (int v : graph[u]) {
            if (inCycle[v]) {
                if (--degree[v] == 1) {
                    inCycle[v] = false;
                    q.push(v);
                }
            }
        }
    }
    return inCycle;
}

vi directional_cycle_vector(const vvi& out_graph) {
    int n = out_graph.size();
    vvi in_graph(n);
    vi in_deg(n, 0), out_deg(n, 0), in_cycle(n, true);
    for (int u = 0; u < n; u++) {
        out_deg[u] = out_graph[u].size();
        for (int v : out_graph[u]) {
            in_graph[v].push_back(u);
            in_deg[v]++;
        }
    }
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (in_deg[i] == 0 || out_deg[i] == 0) {
            q.push(i);
            in_cycle[i] = false;
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : out_graph[u]) {
            if (in_cycle[v]) {
                if (--in_deg[v] == 0) {
                    in_cycle[v] = false;
                    q.push(v);
                }
            }
        }
        for (int v : in_graph[u]) {
            if (in_cycle[v]) {
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
vi diameter_vector(const vt<vt<T>>& graph) {
    int n = graph.size();
    vi d(n);
    pii now = {-1, -1};
    auto dfs = [&](auto& dfs, int node = 0, int par = -1, int depth = 0) -> void {
        if(depth > now.ff) now = MP(depth, node);
        for(auto& nei : graph[node]) {
            if(nei == par) continue;
            dfs(dfs, nei, node, depth + 1);
        }
    }; dfs(dfs);
    int a = now.ss;
    now = {-1, -1};
    dfs(dfs, a);
    int b = now.ss;
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
    auto dp1 = bfs(a), dp2 = bfs(b);
    vi ans(n);
    for(int i = 0; i < n; i++) {
        ans[i] = max(dp1[i], dp2[i]);
    }
    return ans;
}

// Warning: when choosing flow_t, make sure it can handle the sum of flows, not just individual flows.
template<typename flow_t>
struct dinic {
    struct edge {
        int node, _rev;
        flow_t capacity;
 
        edge() {}
 
        edge(int _node, int _rev, flow_t _capacity) : node(_node), _rev(_rev), capacity(_capacity) {}
    };
 
    int V = -1;
    vt<vt<edge>> adj;
    vi dist, edge_index;
    bool flow_called;
 
    dinic(int vertices = -1) {
        if (vertices >= 0)
            init(vertices);
    }
 
    void init(int vertices) {
        V = vertices;
        adj.assign(V, {});
        dist.resize(V);
        edge_index.resize(V);
        flow_called = false;
    }
 
    int _add_edge(int u, int v, flow_t capacity1, flow_t capacity2) {
        assert(0 <= u && u < V && 0 <= v && v < V);
        assert(capacity1 >= 0 && capacity2 >= 0);
        edge uv_edge(v, int(adj[v].size()) + (u == v ? 1 : 0), capacity1);
        edge vu_edge(u, int(adj[u].size()), capacity2);
        adj[u].push_back(uv_edge);
        adj[v].push_back(vu_edge);
        return adj[u].size() - 1;
    }
 
    int add_directional_edge(int u, int v, flow_t capacity) {
        return _add_edge(u, v, capacity, 0);
    }
 
    int add_bidirectional_edge(int u, int v, flow_t capacity) {
        return _add_edge(u, v, capacity, capacity);
    }
 
    edge &reverse_edge(const edge &e) {
        return adj[e.node][e._rev];
    }
 
    void bfs_check(queue<int> &q, int node, int new_dist) {
        if (new_dist < dist[node]) {
            dist[node] = new_dist;
            q.push(node);
        }
    }
 
    bool bfs(int source, int sink) {
        dist.assign(V, inf);
        queue<int> q;
        bfs_check(q, source, 0);
        while (!q.empty()) {
            int top = q.front(); q.pop();
            for (edge &e : adj[top])
                if (e.capacity > 0)
                    bfs_check(q, e.node, dist[top] + 1);
        }
 
        return dist[sink] < inf;
    }
 
    flow_t dfs(int node, flow_t path_cap, int sink) {
        if (node == sink)
            return path_cap;
 
        if (dist[node] >= dist[sink])
            return 0;
 
        flow_t total_flow = 0;
 
        // Because we are only performing DFS in increasing order of dist, we don't have to revisit fully searched edges
        // again later.
        while (edge_index[node] < int(adj[node].size())) {
            edge &e = adj[node][edge_index[node]];
 
            if (e.capacity > 0 && dist[node] + 1 == dist[e.node]) {
                flow_t path = dfs(e.node, min(path_cap, e.capacity), sink);
                path_cap -= path;
                e.capacity -= path;
                reverse_edge(e).capacity += path;
                total_flow += path;
            }
 
            // If path_cap is 0, we don't want to increment edge_index[node] as this edge may not be fully searched yet.
            if (path_cap == 0)
                break;
 
            edge_index[node]++;
        }
 
        return total_flow;
    }
 
    flow_t flow(int source, int sink) {
        assert(V >= 0);
        flow_t total_flow = 0;
 
        while (bfs(source, sink)) {
            edge_index.assign(V, 0);
            total_flow += dfs(source, inf, sink);
        }
 
        flow_called = true;
        return total_flow;
    }
 
    vector<bool> reachable;
 
    void reachable_dfs(int node) {
        reachable[node] = true;
 
        for (edge &e : adj[node])
            if (e.capacity > 0 && !reachable[e.node])
                reachable_dfs(e.node);
    }
 
    // Returns a list of {capacity, {from_node, to_node}} representing edges in the min cut.
    // TODO: for bidirectional edges, divide the resulting capacities by two.
    vector<pair<flow_t, pair<int, int>>> min_cut(int source) {
        assert(flow_called);
        reachable.assign(V, false);
        reachable_dfs(source);
        vector<pair<flow_t, pair<int, int>>> cut;
 
        for (int node = 0; node < V; node++)
            if (reachable[node])
                for (edge &e : adj[node])
                    if (!reachable[e.node])
                        cut.emplace_back(reverse_edge(e).capacity, make_pair(node, e.node));
 
        return cut;
    }
	
	vt<vt<flow_t>> assign_flow(int n) {
        vt<vt<flow_t>> assign(n, vt<flow_t>(n));   
        for(int i = 0; i < n; i++) {
            for(auto& it : adj[i]) {
                int j = it.node - n;
                auto e = reverse_edge(it);
                if(j >= 0 && j < n) {
                    assign[i][j] = e.capacity;
                }
            }
        }
        return assign;
    }
	
	vvi construct_path(int n, vi& a) {
        vi vis(n), A;
        vvi ans, G(n);

        auto dfs = [&](auto& dfs, int node) -> void {
            vis[node] = true;
            A.pb(node + 1); 
            for(auto& nei : G[node]) {
                if(!vis[nei]) {
                    dfs(dfs, nei);
                    return;
                }
            }
        };
        for(int i = 0; i < n; i++) {
            if(a[i] % 2 == 0) continue; // should only add node where going from source to this
            for(auto& it : adj[i]) {
                int j = it.node;
                if(j < n && it.capacity == 0) {
                    G[i].pb(j);
                    G[j].pb(i);
                }
            }
        }
        for(int i = 0; i < n; i++) {
            if(vis[i]) continue;
            A.clear();
            dfs(dfs, i);
            ans.pb(A);
        }
        return ans;
    }

};

struct MCMF {
    public:
    int V;
    struct Edge {
        int to, _rev;
        ll capacity, cost;
        Edge() {}

        Edge(int to, int _rev, ll capacity, ll cost) : to(to), _rev(_rev), capacity(capacity), cost(cost) {}
    };

    vt<vt<Edge>> graph;
    MCMF(int V) : V(V), graph(V) {}

    void add_edge(int u, int v, ll capacity, ll cost) {
        Edge a(v, int(graph[v].size()), capacity, cost);
        Edge b(u, int(graph[u].size()), 0, -cost);
        graph[u].pb(a);
        graph[v].pb(b);
    }

    pll min_cost_flow(int s, int t, ll max_f) { // negate the sign to make max_cost
        ll flow = 0, flow_cost = 0;
        vll prev_v(V, -1), prev_e(V, -1);
        while(flow < max_f) {
            vll dist(V, INF);
            vb vis(V, false);
            queue<int> q;
            dist[s] = 0;
            q.push(s);
            vis[s] = true;
            while(!q.empty()) {
                auto u = q.front(); q.pop();
                vis[u] = false;
                for(int i = 0; i < graph[u].size(); i++) {
                    auto& e = graph[u][i];
                    if(e.capacity > 0 && dist[e.to] > dist[u] + e.cost) {
                        dist[e.to] = dist[u] + e.cost;
                        prev_v[e.to] = u;
                        prev_e[e.to] = i;
                        q.push(e.to);
                        if(!vis[e.to]) {
                            vis[e.to] = true;
                        }
                    } 
                }
            }
            if(dist[t] == INF) break;
            ll df = max_f - flow;
            int v = t;
            while(v != s) {
                int u = prev_v[v];
                int e_idx = prev_e[v];
                df = min(df, graph[u][e_idx].capacity);
                v = u;
            }
            flow += df;
            flow_cost += df * dist[t];
            v = t;
            while(v != s) {
                int u = prev_v[v];
                int e_idx = prev_e[v];
                graph[u][e_idx].capacity -= df;
                graph[v][graph[u][e_idx]._rev].capacity += df;
                v = u;
            }
        }
        return {flow, flow_cost};
    }
};

class Kuhn {
public:
    int n, l;
    vvi adj;
    vi mate, vis;

    Kuhn(int nn, int _ll)
        : n(nn), l(_ll), adj(nn), mate(nn, -1), vis(nn, false) {}

    void add_edge(int v, int u) {
        adj[v].pb(u);
        adj[u].pb(v);
    }

    bool dfs(int v) {
        if (vis[v]) return false;
        vis[v] = true;
        for (int w : adj[v]) {
            if (!vis[w]) {
                vis[w] = true;
                if (mate[w] == -1 || dfs(mate[w])) {
                    mate[v] = w;
                    mate[w] = v;
                    return true;
                }
            }
        }
        return false;
    }

    int max_match() {
        int ans = 0;
        while (true) {
            fill(vis.begin(), vis.end(), false);
            bool aug = false;
            for (int i = 0; i < l; i++) {
                if (mate[i] == -1 && !vis[i] && dfs(i)) {
                    aug = true;
                    ans++;
                }
            }
            if (!aug) break;
        }
        return ans;
    }
};

struct Blossom {
    int n;
    vi match, Q, pre, base, hash, in_blossom, in_path;
    vvi adj;
    Blossom(int n) : n(n), match(n, -1), adj(n, vi(n)), hash(n), Q(n), pre(n), base(n), in_blossom(n), in_path(n) {}

    void insert(const int &u, const int &v) {
        adj[u][v] = adj[v][u] = 1;
    }

    int max_match() {
        fill(all(match), -1);
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            if (match[i] == -1) ans += bfs(i);
        }
        return ans;
    }

    int bfs(int p) {
        fill(all(pre), -1);
        fill(all(hash), 0);
        iota(all(base), 0);
        Q[0] = p;
        hash[p] = 1;
        for (int s = 0, t = 1; s < t; ++s) {
            int u = Q[s];
            for (int v = 0; v < n; ++v) {
                if (adj[u][v] && base[u] != base[v] && v != match[u]) {
                    if (v == p || (match[v] != -1 && pre[match[v]] != -1)) {
                        int b = contract(u, v);
                        for (int i = 0; i < n; ++i) {
                            if (in_blossom[base[i]]) {
                                base[i] = b;
                                if (hash[i] == 0) {
                                    hash[i] = 1;
                                    Q[t++] = i;
                                }
                            }
                        }
                    } else if (pre[v] == -1) {
                        pre[v] = u;
                        if (match[v] == -1) {
                            argument(v);
                            return 1;
                        } else {
                            Q[t++] = match[v];
                            hash[match[v]] = 1;
                        }
                    }
                }
            }
        }
        return 0;
    }

    void argument(int u) {
        while (u != -1) {
            int v = pre[u];
            int k = match[v];
            match[u] = v;
            match[v] = u;
            u = k;
        }
    }

    void change_blossom(int b, int u) {
        while (base[u] != b) {
            int v = match[u];
            in_blossom[base[v]] = in_blossom[base[u]] = true;
            u = pre[v];
            if (base[u] != b) {
                pre[u] = v;
            }
        }
    }

    int contract(int u, int v) {
        fill(all(in_blossom), 0);
        int b = find_base(base[u], base[v]);
        change_blossom(b, u);
        change_blossom(b, v);
        if (base[u] != b) pre[u] = v;
        if (base[v] != b) pre[v] = u;
        return b;
    }

    int find_base(int u, int v) {
        fill(all(in_path), 0);
        while (true) {
            in_path[u] = true;
            if (match[u] == -1) {
                break;
            }
            u = base[pre[match[u]]];
        }
        while (!in_path[v]) {
            v = base[pre[match[v]]];
        }
        return v;
    }
};

template <class T, T oo>
struct HopcroftKarp {
    int n, m; 
    vvi adj;
    vi pairU, pairV;
    vt<T> dist;

    HopcroftKarp(int n, int m) : n(n), m(m) {
        adj.resize(n);
        pairU.assign(n, -1);
        pairV.assign(m, -1);
        dist.assign(n, oo);
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
    }

    bool bfs() {
        queue<int> q;
        for (int u = 0; u < n; u++) {
            if (pairU[u] == -1) {
                dist[u] = 0;
                q.push(u);
            } else {
                dist[u] = oo;
            }
        }
        T INF = oo;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            if (dist[u] < INF) {
                for (int v : adj[u]) {
                    if (pairV[v] == -1) {
                        INF = dist[u] + 1;
                    } else if (dist[pairV[v]] == oo) {
                        dist[pairV[v]] = dist[u] + 1;
                        q.push(pairV[v]);
                    }
                }
            }
        }
        return INF != oo;
    }

    bool dfs(int u) {
        if (u != -1) {
            for (int v : adj[u]) {
                int pu = pairV[v];
                if (pu == -1 || (dist[pu] == dist[u] + 1 && dfs(pu))) {
                    pairV[v] = u;
                    pairU[u] = v;
                    return true;
                }
            }
            dist[u] = oo;
            return false;
        }
        return true;
    }

    int max_match() {
        int matching = 0;
        while (bfs()) {
            for (int u = 0; u < n; u++) {
                if (pairU[u] == -1 && dfs(u)) {
                    matching++;
                }
            }
        }
        return matching;
    }
	
	vpii getMatching() const {
        vpii matchingPairs;
        for (int u = 0; u < n; u++) {
            if (pairU[u] != -1) {
                matchingPairs.push_back({u, pairU[u]});
            }
        }
        return matchingPairs;
    }

};

template<class T, T oo>
struct Hungarian {
    int n, m;
    vi maty, frm, used;
    vt<vt<T>> cst;
    vt<T> fx, fy, dst;

    Hungarian(int n, int m) {
        this->n = n;
        this->m = m;
        cst.resize(n + 1, vt<T>(m + 1, oo));
        fx.resize(n + 1);
        fy.resize(m + 1);
        dst.resize(m + 1);
        maty.resize(m + 1);
        frm.resize(m + 1);
        used.resize(m + 1);
    }

    void add_edge(int x, int y, T c) {
        cst[x][y] = c;
    }

    T min_cost() {
        random_device rd;
        mt19937 rng(rd());
        for (int x = 1; x <= n; x++) {
            int y0 = 0;
            maty[0] = x;
            for (int y = 0; y <= m; y++) {
                dst[y] = oo + 1;
                used[y] = 0;
            }
            int y1;
            do {
                used[y0] = 1;
                int x0 = maty[y0];
                T delta = oo + 1;
                vi perm(m);
                for (int i = 0; i < m; i++) {
                    perm[i] = i + 1;
                }
                shuffle(perm.begin(), perm.end(), rng);
                for (int idx = 0; idx < m; idx++) {
                    int y = perm[idx];
                    if (!used[y]) {
                        T curdst = cst[x0][y] - fx[x0] - fy[y];
                        if (dst[y] > curdst) {
                            dst[y] = curdst;
                            frm[y] = y0;
                        }
                        if (delta > dst[y]) {
                            delta = dst[y];
                            y1 = y;
                        }
                    }
                }
                for (int y = 0; y <= m; y++) {
                    if (used[y]) {
                        fx[maty[y]] += delta;
                        fy[y] -= delta;
                    } else {
                        dst[y] -= delta;
                    }
                }
                y0 = y1;
            } while (maty[y0] != 0);
            do {
                int y1 = frm[y0];
                maty[y0] = maty[y1];
                y0 = y1;
            } while (y0);
        }
        T res = 0;
        for (int y = 1; y <= m; y++) {
            T x = maty[y];
            if (cst[x][y] < oo)
                res += cst[x][y];
        }
        return res;
    }
};
