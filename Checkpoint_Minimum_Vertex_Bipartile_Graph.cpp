#include <bits/stdc++.h>
using namespace std;

#define int long long
const int INF = 1e12;

int n, m, p;
vector<vector<int>> adj;        // Adjacency list for the graph
vector<int> leftMatch, rightMatch;  // Matchings for left and right sides
vector<int> dist;               // Distances used for BFS

// BFS to check if there is an augmenting path
bool bfs() {
    queue<int> q;

    for (int i = 1; i <= n; i++) {
        if (leftMatch[i] == 0) {
            dist[i] = 0;
            q.push(i);
        } else {
            dist[i] = INF;
        }
    }

    dist[0] = INF;

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        if (dist[node] < dist[0]) {
            for (int neighbor : adj[node]) {
                if (dist[rightMatch[neighbor]] == INF) {
                    dist[rightMatch[neighbor]] = dist[node] + 1;
                    q.push(rightMatch[neighbor]);
                }
            }
        }
    }

    return dist[0] != INF;
}

// DFS to find and augment paths
bool dfs(int node) {
    if (node == 0) {
        return true;
    }

    for (int neighbor : adj[node]) {
        if (dist[rightMatch[neighbor]] == dist[node] + 1) {
            if (dfs(rightMatch[neighbor])) {
                rightMatch[neighbor] = node;
                leftMatch[node] = neighbor;
                return true;
            }
        }
    }

    dist[node] = INF;
    return false;
}

void solve() {
    cin >> n >> m >> p;

    adj.resize(n + 1);
    leftMatch.assign(n + 1, 0);
    rightMatch.assign(m + 1, 0);
    dist.assign(n + 1, INF);

    for (int i = 0; i < p; i++) {
        int x, y;
        cin >> x >> y;
        adj[x].push_back(y);
    }

    int maxMatching = 0;

    while (bfs()) {
        for (int i = 1; i <= n; i++) {
            if (leftMatch[i] == 0 && dfs(i)) {
                maxMatching++;
            }
        }
    }

    cout << maxMatching << endl;
}

int32_t main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    solve();
    return 0;
}

