class GRAPH { 
    public: 
    int n;  
    vvi dp, graph; 
    vi depth, parent;
    vi startTime, endTime, low, tin;
	vi subtree;
    int timer = 0, centroid1 = -1, centroid2 = -1, mn = inf, diameter = 0;
    GRAPH(vvi& graph) {   
        this->graph = graph;
        n = graph.size();
        dp.rsz(n, vi(MK));
        depth.rsz(n);
        parent.rsz(n, 1);
        startTime.rsz(n);   
        endTime.rsz(n);
        subtree.rsz(n);
		low.rsz(n);
		tin.rsz(n);
        dfs();
        init();
    }
    
    int dfs(int node = 0, int par = -1) {   
        startTime[node] = timer++;
		subtree[node] = 1;
        int mx = 0, a = 0, b = 0;
        for(auto& nei : graph[node]) {  
            if(nei == par) continue;    
            depth[nei] = depth[node] + 1;   
            dp[nei][0] = node;
            parent[nei] = node;
			int v = dfs(nei, node);
            if(v > a) b = a, a = v; 
            else b = max(b, v);
			subtree[node] += subtree[nei];
			mx = max(mx, subtree[nei]);
        }
		diameter = max(diameter, a + b); // might be offset by 1
        endTime[node] = timer - 1;
        mx = max(mx, n - subtree[node] - 1); // careful with offSet, may take off -1
		if(mx < mn) mn = mx, centroid1 = node, centroid2 = -1;
		else if(mx == mn) centroid2 = node;
		return a + 1;
    }
    
    void init() {  
        for(int j = 1; j < MK; j++) {   
            for(int i = 0; i < n; i++) {    
                dp[i][j] = dp[dp[i][j - 1]][j - 1];
            }
        }
    }
    
    bool isAncestor(int u, int v) { 
        return startTime[u] <= startTime[v] && startTime[v] <= endTime[u]; 
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
            if((k >> i) & 1) {  
                a = dp[a][i];
            }
            if(a == 0) return -1;
        }
        return a;
    }
	
	void bridge_dfs(int node = 0, int par = -1) {
        low[node] = tin[node] = timer++; 
        subtree[node] = 1;
        for(auto& nei : graph[node]) {  
            if(nei == par) continue;
            if(!tin[nei]) {   
                bridge_dfs(nei, node);
                subtree[node] += subtree[nei];
                low[node] = min(low[node], low[nei]);   
                if(low[nei] > tin[node]) {  
                    //res = max(res, (ll)subtree[nei] * (n - subtree[nei]));
                }
            }
            else {  
                low[node] = min(low[node], tin[nei]);
            }
        }
    };

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

struct line {
    ll m, b;
    mutable function<const line*()> succ;
    bool operator<(const line& rhs) const {
        if (rhs.b != -INF) return m < rhs.m;
        const line* s = succ();
        if (!s) return 0;
        ll x = rhs.m;
        return b - s->b < (s->m - m) * x;
    }
};
 
struct CHT : public multiset<line> { // will maintain upper hull for maximum
    // do update in this form : a + mx -> insert_line(m, -a)
    // do queries in this form : x - queries(condition)
    // example : dp[i] = dp[j] + i * j
    // update : insert_line(j, -dp[j])
    // queries : dp[i] = cht.queries(i)

    bool bad(iterator y) {
        auto z = next(y);
        if (y == begin()) {
            if (z == end()) return 0;
            return y->m == z->m && y->b <= z->b;
        }
        auto x = prev(y);
        if (z == end()) return y->m == x->m && y->b <= x->b;
 
		/* compare two lines by slope, make sure denominator is not 0 */
        ll v1 = (x->b - y->b);
        if (y->m == x->m) v1 = x->b > y->b ? INF : -INF;
        else v1 /= (y->m - x->m);
        ll v2 = (y->b - z->b);
        if (z->m == y->m) v2 = y->b > z->b ? INF : -INF;
        else v2 /= (z->m - y->m);
        return v1 >= v2;
    }
    void insert_line(ll m, ll b) {
        auto y = insert({ m, b });
        y->succ = [this, y] {
            return next(y) == end() ? nullptr : &*next(y);
        };
        if (bad(y)) { erase(y); return; }
        while (next(y) != end() && bad(next(y))) erase(next(y));
        while (y != begin() && bad(prev(y))) erase(prev(y));
    }
    ll queries(ll x) {
        auto l = *lower_bound((line) { x, -INF });
        return l.m * x + l.b;
    }
};
