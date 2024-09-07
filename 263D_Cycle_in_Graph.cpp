#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class DSU {
public:
    vector<int> root, size;
    int n;
    
    DSU(int n) {
        this->n = n;
        root.resize(n, -1);
        size.resize(n, 1);
    }
    
    int find(int x) {
        if (root[x] == -1) return x;
        return root[x] = find(root[x]);
    }
    
    void merge(int u, int v) {
        u = find(u), v = find(v);
        if (u != v) {
            if (size[u] < size[v]) swap(u, v);
            root[v] = u;
            size[u] += size[v];
        }
    }
    
    void findComponentWithSizeAtLeast(int k) {
        for (int i = 0; i < n; i++) {
            if (size[find(i)] > k) {
                int rootNode = find(i);
                cout << size[rootNode] << endl;
                for (int j = 0; j < n; j++) {
                    if (find(j) == rootNode) {
                        cout << j + 1 << " ";
                    }
                }
                cout << endl;
                return;
            }
        }
    }
};

void solve() {
    int n, m, k;
    cin >> n >> m >> k;
    
    vector<vector<int>> graph(n);
    vector<int> degree(n);
    
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        a--, b--;
        graph[a].push_back(b);
        graph[b].push_back(a);
        degree[a]++;
        degree[b]++;
    }
    
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) q.push(i);
    }
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        for (int nei : graph[node]) {
            if (--degree[nei] == 1) {
                q.push(nei);
            }
        }
    }
    
    DSU dsu(n);
    for (int i = 0; i < n; i++) {
        if (degree[i] > 1) {
            for (int nei : graph[i]) {
                if (degree[nei] > 1) {
                    dsu.merge(i, nei);
                }
            }
        }
    }
    
    dsu.findComponentWithSizeAtLeast(k);
}

int main() {
    solve();
    return 0;
}

