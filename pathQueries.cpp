// https://cses.fi/problemset/task/1138/
#include <ext/pb_ds/assoc_container.hpp>
#include <bits/stdc++.h>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

class FenwickTree
{
    public:
    int n;
    vector<ll> root;
    FenwickTree(int n)
    {
        this->n = n;
        root.resize(n + 1);
    }

    void update(int id, ll val)
    {
        while(id <= n)
        {
            root[id] += val;
            id += (id & -id);
        }
    }

    ll get(int id)
    {
        ll res = 0;
        while(id)
        {
            res += root[id];
            id -= (id & -id);
        }
        return res;
    }
};


class SegmentTree
{
    public:
    int n;
    vector<ll> root;
    SegmentTree(int n)
    {
        this->n = n;
        root.resize(n * 4);
    }

    void update(int index, int val)
    {
        update(0, 0, n - 1, index, val);
    }

    void update(int i, int left, int right, int index, int val)
    {
        if(left == right)
        {
            root[i] = val;
            return;
        }
        int middle = left + (right - left) / 2;
        if(index <= middle) update(i * 2 + 1, left, middle, index, val);
        else update(i * 2 + 2, middle + 1, right, index, val);
        root[i] = root[i * 2 + 1] + root[i * 2 + 2];
    }

    ll get(int start, int end)
    {
        return get(0, 0, n - 1, start, end);
    }

    ll get(int i, int left, int right, int start, int end)
    {
        if(left >= start && right <= end) return root[i];
        if(start > right || left > end) return 0;
        int middle = left + (right - left) / 2;
        return get(i * 2 + 1, left, middle, start, end) + get(i * 2 + 2, middle + 1, right, start, end);
    }
};

vector<vector<int>> graph;
vector<int> values, startTime, endTime;
int currTime = 0;

void dfs(int node, int par)
{
    startTime[node] = ++currTime;
    for(auto& nei : graph[node])
    {
        if(nei == par) continue;
        dfs(nei, node);
    }
    endTime[node] = ++currTime;
} 

void solve()
{
    int n, q;
    cin >> n >> q;
    graph.resize(n), startTime.resize(n), endTime.resize(n), values.resize(n);
    for(auto& it : values) cin >> it;

    for(int i = 0; i < n - 1; i++)
    {
        int a, b;
        cin >> a >> b;
        a--, b--;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    dfs(0, -1);
//    FenwickTree root(currTime);
    SegmentTree root(currTime);
    for(int i = 0; i < n; i++)
    {
//        root.update(startTime[i], values[i]);
        root.update(startTime[i], values[i]);
        root.update(endTime[i], -values[i]);
    }
   
    while(q--)
    {
        int a, b;
        cin >> a >> b;
        b--;
        if(a == 1)
        {
            int c;
            cin >> c;
            root.update(startTime[b], c);
            root.update(endTime[b], -c);
        }
        else
        {
            cout << root.get(1, startTime[b]) << endl;
        }
    }

}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

