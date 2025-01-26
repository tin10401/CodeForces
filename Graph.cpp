class GRAPH { 
    public: 
    int n;  
    vvi dp, graph; 
    vi depth, parent, subtree;
    vi tin, tout, low;
    int timer = 0;
//    int centroid1 = -1, centroid2 = -1, mn = inf, diameter = 0;
    GRAPH(vvi& graph, int root = 0) {   
        this->graph = graph;
        n = graph.size();
        dp.rsz(n, vi(MK));
        depth.rsz(n);
        parent.rsz(n, -1);
		subtree.rsz(n, 1);
//        tin.rsz(n);
//        tout.rsz(n);
//        low.rsz(n);
        dfs(root);
        init();
    }
    
    void dfs(int node = 0, int par = -1) {   
        for(auto& nei : graph[node]) {  
            if(nei == par) continue;    
            depth[nei] = depth[node] + 1;   
            dp[nei][0] = node;
            parent[nei] = node;
			dfs(nei, node);
			subtree[node] += subtree[nei];
        }
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
//        for(int j = 1; j < MK; j++) {
//            int p = dp[u][j - 1];
//            dp[u][j] = dp[p][j - 1];
//            ans[u][j] = merge(ans[u][j - 1], ans[p][j - 1]);
//        }
//    }

    bool isAncestor(int u, int v) { 
        return tin[u] <= tin[v] && tin[v] <= tout[u]; 
    }
    
    void init() {  
        for(int j = 1; j < MK; j++) {   
            for(int i = 0; i < n; i++) {    
                dp[i][j] = dp[dp[i][j - 1]][j - 1];
            }
        }
    }
	
    int lca(int a, int b) { 
        if(depth[a] > depth[b]) {   
            swap(a, b);
        }
        int d = depth[b] - depth[a];    
        for(int i = MK - 1; i >= 0; i--) {  
            if((d >> i) & 1) {  
                b = dp[b][i];
            }
        }
        if(a == b) return a;    
        for(int i = MK - 1; i >= 0; i--) {  
            if(dp[a][i] != dp[b][i]) {  
                a = dp[a][i];   
                b = dp[b][i];
            }
        }
        return dp[a][0];
    }
	
	int dist(int u, int v) {    
        int a = lca(u, v);  
        return depth[u] + depth[v] - 2 * depth[a];
    }
	
	int k_ancestor(int a, int k) {
        for(int i = MK - 1; i >= 0; i--) {   
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

class DSU { 
    public: 
    int n;  
    vi root, rank;  
    DSU(int n) {    
        this->n = n;    
        root.rsz(n, -1), rank.rsz(n, 1);
    }
    
    int find(int x) {   
        if(root[x] == -1) return x; 
        return root[x] = find(root[x]);
    }
    
    bool merge(int u, int v) {  
        u = find(u), v = find(v);   
        if(u != v) {    
            if(rank[v] > rank[u]) swap(u, v);   
            rank[u] += rank[v]; 
            root[v] = u;
            return true;
        }
        return false;
    }
    
    bool same(int u, int v) {    
        return find(u) == find(v);
    }
    
    int getRank(int x) {    
        return rank[find(x)];
    }
};

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
		if(rank[u].back().ss < rank[v].back().ss) swap(u, v);

		version = ver;
		int szu = rank[u].back().ss;
		int szv = rank[v].back().ss;
		if (szu > szv) {swap(u, v);}
		parent[u].pb({version, v});
		int new_sz = szu + szv;
		rank[v].pb({version, new_sz});
		return version;
	}
 
	bool same(int u, int v, int ver) {
        return find(u, ver) == find(v, ver);
	}
};

class Undo_DSU {
    public:
    vi par, rank;
    stack<ar(4)> st;
    int n;
    int comp;
    ll res;
    Undo_DSU(int n) {
        this->n = n;
        this->comp = n;
        res = 0;
        par.rsz(n), rank.rsz(n, 1);
        iota(all(par), 0);
    }
 
    int find(int v) {
        if (par[v] == v) return v;
        return find(par[v]);
    }
 
    bool merge(int a, int b, bool save = false) {
        a = find(a); b = find(b);
        if (a == b) return false;
        comp--;
        if (rank[a] < rank[b]) swap(a, b);
        if (save) st.push({a, rank[a], b, rank[b]});
        ll v = 1LL * rank[a] * rank[b];
        res += v;
        par[b] = a;
        rank[a] += rank[b];
        return true;
    }
 
    void rollBack() {
        if(!st.empty()) {
            comp++;
            auto x = st.top(); st.pop();
            ll v = 1LL * x[1] * x[3];
            res -= v;
            par[x[0]] = x[0];
            rank[x[0]] = x[1];
            par[x[2]] = x[2];
            rank[x[2]] = x[3];
        }
    }
 
    bool same(int u, int v) {
        return find(u) == find(v);
    }
 
    int getRank(int u) {
        return rank[find(u)];
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
		// don't forget to generate after adding edges
    }
 
    void add(int a, int b) {    
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
 
    void generate() {
        for(int i = 0; i < n; i++) dfs(i);
        while(!s.empty()) {
            int node = s.top(); s.pop();
            if(comp[node] != -1) continue;
            dfs2(node);
            curr_comp++;
        }
    }
    
    vvi condense() {    
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

class CHT {
    public:
    int is_mx;
    vll m_slopes, b_intercepts;
    CHT(int is_mx) : is_mx(is_mx) {
        add_line(0, 0);
    }

    db cross(int i, int j, int k) {
        db A = (db)(1.00 * m_slopes[j] - m_slopes[i]) * (b_intercepts[k] - b_intercepts[i]);
        db B = (db)(1.00 * m_slopes[k] - m_slopes[i]) * (b_intercepts[j] - b_intercepts[i]);
        return is_mx ? A < B : A >= B;
    }

    void add(ll a, ll b) {
        if(is_mx) add_line(a, -b);
        else add_line(-a, b);
    }

    ll queries(ll x) {
        return is_mx ? -get(x) : get(x);
    }

    void add_line(ll slope, ll intercept) {
        m_slopes.push_back(slope);
        b_intercepts.push_back(intercept);
        while(m_slopes.size() >= 3 && cross(m_slopes.size() - 3, m_slopes.size() - 2, m_slopes.size() - 1)) {
            m_slopes.erase(m_slopes.end() - 2);
            b_intercepts.erase(b_intercepts.end() - 2);
        }
    }

    ll get(ll x) {
        if(m_slopes.empty()) return INF;
        int l = 0, r = m_slopes.size() - 1;
        while(l < r) {
            int mid = l + (r - l) / 2;
            ll f1 = m_slopes[mid] * x + b_intercepts[mid];
            ll f2 = m_slopes[mid + 1] * x + b_intercepts[mid + 1];
            if(f1 > f2) l = mid + 1;
            else r = mid;
        }
        return m_slopes[l] * x + b_intercepts[l];
    }
};

struct Line {
    mutable ll m, c, p;
    bool isQuery;
    bool operator<(const Line& o) const {
        if(o.isQuery)
            return p < o.p;
        return m < o.m;
    }
};

struct CHT : multiset<Line> {
    const ll inf = INF;
    int is_mx;
    ll div(ll a, ll b) {
        return a / b - ((a ^ b) < 0 && a % b); }
    bool isect(iterator x, iterator y) {
        if (y == end()) { x->p = inf; return false; }
        if (x->m == y->m) x->p = x->c > y->c ? inf : -inf;
        else x->p = div(y->c - x->c, x->m - y->m);
        return x->p >= y->p;
    }
    void add(ll m, ll c) {
        auto z = insert({m, c, 0, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p)
            isect(x, erase(y));
    }
    ll query(ll x) {
        if(empty()) return inf;
        Line q; q.p = x, q.isQuery = 1;
        auto l = *lower_bound(q);
        return l.m * x + l.c;
    }
    // min will return -ans;
    // max will return ans;
    // max_normall is add(i, -dp)
    // min_normal is add(-i, dp)
};

struct CD { // centroid_decomposition
    int n, root;
    vvi graph;
    vi size, parent, vis, a, freq;
    vll ans;
    CD(vvi& graph, vi& a) : graph(graph), n(graph.size()), a(a) {
        size.rsz(n);
        parent.rsz(n);
        vis.rsz(n);
        freq.rsz((1 << MK) + 20);
        freq[0] = 1;
        ans.rsz(n);
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

    int get(int mask) {
        int res = freq[mask];
        for(int i = 0; i < MK; i++) {
            res += freq[mask ^ (1 << i)];
        }
        return res;
    }

    void modify(int node, int par, int mask, int delta) {
        mask ^= a[node]; 
        freq[mask] += delta;
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            modify(nei, node, mask, delta);
        }
    }
 
    ll cal(int node, int par, int mask) {
        mask ^= a[node];
        ll res = get(mask);
        for(auto& nei : graph[node]) {
            if(nei == par || vis[nei]) continue;
            res += cal(nei, node, mask);
        }
        ans[node] += res;
        return res;
    }
	
	int get_max_depth(int node, int par, int d, int last) {
        if(d > k) return 0;
        int curr = d;
        for(auto& [nei, t] : graph[node]) {
            if(nei == par || vis[nei]) continue;
            curr = max(curr, get_max_depth(nei, node, d +  int(last != -1 && last != t), t)); 
        }
        return curr;
    }

    int init(int root = 0, int par = -1) {
        root = get_centroid(root);
        parent[root] = par;
        for(auto& nei : graph[root]) {
            if(nei == par || vis[nei]) continue;
            modify(nei, root, 0, 1);
        }
        ll sm = 1;
        for(auto& nei : graph[root]) {
            if(nei == par || vis[nei]) continue;
            modify(nei, root, 0, -1);
            sm += cal(nei, root, a[root]);
            modify(nei, root, 0, 1);
        }
        sm += get(a[root]);
        sm >>= 1;
        ans[root] += sm;
        for(auto& nei : graph[root]) {
            if(nei == par || vis[nei]) continue;
            modify(nei, root, 0, -1);
        }
        for(auto&nei : graph[root]) {
            if(nei == par || vis[nei]) continue;
            init(nei, root);
        }
        return root;
    }
};

